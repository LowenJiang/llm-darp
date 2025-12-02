import re
import os
from ast import literal_eval
import math
import json
import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

llm_client = OpenAI()
N = 30
K = 50

REGENERATE_TRIPS = False
REGENERATE_DECISIONS = True

# ============================
# Helper Functions
# ============================

def safe_json_parse(text):
    """
    Tries to extract and parse JSON list from LLM output robustly.
    """
    if text is None:
        return []
    try:
        # Try direct parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract JSON substring
        json_str = re.search(r'\[.*\]', text, re.DOTALL)
        if json_str:
            json_str = json_str.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Clean up minor formatting issues
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*\]', ']', json_str)
                try:
                    return json.loads(json_str)
                except Exception:
                    return []
        else:
            return []
        
# ============================
# STEP 1: Generate Static Persona Features
# ============================

ages = ["18-25", "26-35", "36-50", "51-65", "65+"]
#genders = ["male", "female", "non-binary", "prefer not to say"]
#racial_identities = ["White", "Black or African American", "Asian", "Hispanic or Latino", "Native American", "Other"]
occupations = ["student", "teacher", "healthcare worker", "IT professional", "construction worker", "retired", "unemployed"]
incomes = ["Less than $25K", "$25K - $50K", "$50K - $75K", "$75K - $100K", "$100K - $150K", "$150K or more"]
time_flexibilities = ["flexible for both early pickup and late dropoff", "flexible for early pickup, but inflexible for late dropoff", "flexible for late dropoff, but inflexible for early pickup", "inflexible for any schedule changes"]

def generate_traveler_profiles(N):
    travelers = []
    for i in range(N):
        traveler = {
            "traveler_id": i + 1,
            "age": np.random.choice(ages),
#            "gender": np.random.choice(genders),
#            "racial_identity": np.random.choice(racial_identities),
            "occupation": np.random.choice(occupations),
            "income": np.random.choice(incomes),
            "flexible": np.random.choice(time_flexibilities)
        }
        travelers.append(traveler)
    return pd.DataFrame(travelers)

traveler_df = generate_traveler_profiles(N)

# ============================
# STEP 2: Ask OpenAI API to Generate Trip Types per Traveler
# ============================

def generate_trip_types(traveler, K=3):
    """
    Use the OpenAI API to generate K trip types consistent with a traveler's persona.
    """
    persona_description = (
        f"The traveler is {traveler['age']} years old, "
        f"works as a {traveler['occupation']} with an income level of {traveler['income']}."
        f"The traveler is {traveler['flexible']}."
    )

    prompt = (
        f"{persona_description}\n\n"
        f"Generate {K} trip types this traveler is likely to request in a dial-a-ride system set in San Francisco. "
        f"For each trip type, provide the following features in a JSON list format:\n"
        f"- trip_purpose\n"
        f"- departure_location\n"
        f"- arrival_location\n"
        f"- origin_gps (latitude, longitude)\n"
        f"- destination_gps (latitude, longitude)\n"
        f"- trip_distance_miles (realistic driving distance, consistent with Google Maps results for these GPS coordinates)\n"
        f"- trip_duration_minutes (realistic driving time under typical San Francisco traffic, consistent with Google Maps for the same trip)\n"
        f"- departure_time_window (in 24-hour format, e.g., '07:30-08:00')\n"
        f"- arrival_time_window (in 24-hour format, e.g., '08:00-08:30')\n"
        f"- flexibility_pickup_earlier (maximum minutes the traveler is willing to start the trip earlier than the preferred departure window, e.g., 0, 10, 30)\n"
        f"- flexibility_dropoff_later (maximum minutes the traveler is willing to arrive later than the preferred arrival window, e.g., 0, 10, 30)\n\n"
        f"Requirements for spatial and temporal consistency:\n"
        f"- All GPS points must be within the San Francisco city boundary.\n"
        f"- Use realistic coordinates that match the land-use type:\n"
        f"  - 'home' → residential neighborhood (e.g., Inner Sunset, Richmond, Noe Valley). Please be consistent with this person's income level.\n"
        f"  - 'workplace' → commercial area (e.g., Financial District, SoMa). Please be consistent with this traveler's occupation.\n"
        f"  - 'shopping' → retail/commercial areas (e.g., Union Square, Stonestown Galleria)\n"
        f"  - 'recreation' or 'hiking' → parks (e.g., Golden Gate Park, Presidio, Lands End)\n"
        f"- The same named location (e.g., 'home') must always correspond to the same GPS coordinates for this traveler.\n"
        f"- The trip_distance_miles and trip_duration_minutes must be consistent with the real-world driving distance and time "
        f"you would obtain if querying Google Maps with those GPS coordinates.\n"
        f"  (Example: if the origin is in Noe Valley and destination is in Financial District, expect around 5 miles and 20–25 minutes depending on traffic.)\n"
        f"- The arrival_time_window must be FEASIBLE given the departure_time_window and trip_duration_minutes.\n"
        f"  (Example: if departure is 07:30–08:00 and duration is 20 minutes, arrival should be around 07:50–08:20.)\n\n"
        f"When assigning flexibilities, apply the following logic:\n"
        f"- Let the **trip purpose** be the main driver of flexibility for that trip (e.g., work or school trips are strict; leisure or shopping trips are relaxed).\n"
        f"- Let the **traveler's persona** shape their baseline tendency across all trips "
        f"(e.g., a retiree or freelancer generally shows higher flexibility across trips than a nurse or office worker).\n"
        f"- Make the pickup and dropoff flexibilities potentially different — they do NOT have to be equal.\n\n"
        f"Ensure that all generated trip types are:\n"
        f"1. Geographically realistic (GPS, distances, durations match real-world geography of San Francisco);\n"
        f"2. Temporally consistent (arrival/departure times feasible given duration);\n"
        f"3. Behaviorally coherent with the traveler's persona and daily life patterns.\n\n"
        f"Return only valid JSON, no explanations or comments."
    )

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7#,
#            max_tokens=50
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        print(f"Error generating trips for traveler {traveler['traveler_id']}: {e}")
        return None

# ============================
# STEP 2: Ask OpenAI API to Generate acceptance/rejection for each proposed time shift of each trip record
# ============================
def generate_decisions(trip_record, flexibilities = [], dt = 10, max_shift = 30):
    # Define all possible combinations of pickup/dropoff shifts
    pickup_shifts = list(range(0, max_shift + dt, dt))
    dropoff_shifts = list(range(0, max_shift + dt, dt))
    shift_combinations = list(itertools.product(pickup_shifts, dropoff_shifts))
    
    def get_discount(pickup_shift, dropoff_shift, distance):
        return (abs(pickup_shift) + abs(dropoff_shift)) * distance / 100

    # Create a string describing the combinations (for context in prompt)
    shift_description = "\n".join(
        [f"- pickup_shift_min: {p}, dropoff_shift_min: {d}, discount: {get_discount(p, d, trip_record['trip_distance_miles'])}" for p, d in shift_combinations]
    )
    
    # Combine all flexibility levels in one prompt
    flex_section = "\n".join([
        f"{i+1}. {flex}" for i, flex in enumerate(flexibilities)
    ])

    prompt = f"""
        You are predicting whether a traveler will accept or reject proposed time shifts
        for a flexible pickup/dropoff service. Travelers may receive a fare discount when
        agreeing to shift times, but their willingness depends on persona and trip context.

        Traveler profile:
        - Age: {trip_record['age']}
        - Occupation: {trip_record['occupation']}
        - Annual income: {trip_record['income']}

        Trip context:
        - Purpose: {trip_record['trip_purpose']}
        - From: {trip_record['departure_location']} ({trip_record['origin_gps']})
        - To: {trip_record['arrival_location']} ({trip_record['destination_gps']})
        - Trip distance: {trip_record['trip_distance_miles']} miles
        - Expected duration: {trip_record['trip_duration_minutes']} minutes
        - Preferred departure: {trip_record['departure_time_window']}
        - Preferred arrival: {trip_record['arrival_time_window']}
        
        Below is the list of flexibility levels to consider.
        **You must choose exactly one of these options as the 'flexibility_type' for each prediction.**
        Do not summarize, rephrase, or infer new types — copy the text exactly as it appears.

        {flex_section}

        Below are all possible combinations of pickup and dropoff shifts (in minutes):

        {shift_description}

        For each combination, predict whether the traveler will ACCEPT or REJECT the offer, considering:
        - how large the total shift is relative to their flexibility
        - trip purpose (e.g., work/school strict; leisure/shopping relaxed)
        - traveler persona (income, occupation, age)
        - an assumed proportional fare discount increasing with total shift

        Return only **valid JSON** in the following format:

        {{
          "decisions": [
            {{
              "flexibility_type": <exactly one of the options from the list above>,
              "pickup_shift_min": <minutes earlier>,
              "dropoff_shift_min": <minutes later>,
              "decision": "accept" or "reject"
            }},
            ...
          ]
        }}
        Include one entry for **each combination** of flexibility type and shift pair ({len(flexibilities)} × {len(shift_combinations)} total). Do NOT modify the flexibility_type provided. Do NOT include explanations or comments.
    """

    # Send a single prompt for all shift combinations
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a reasoning assistant that outputs strictly valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    # Parse model output safely
    content = response.choices[0].message.content.strip()
    try:
        parsed = safe_json_parse(content)
    except json.JSONDecodeError:
        print("Warning: Invalid JSON returned. Raw content:")
        print(content)
        parsed = {"decisions": []}

    return parsed["decisions"]

# ============================
# Run for all travelers
# ============================

if REGENERATE_TRIPS:
    all_trips = []

    for i in tqdm(range(traveler_df.shape[0])):
        traveler = traveler_df.iloc[i]
        trips_text = generate_trip_types(traveler, K)
        trip_list = safe_json_parse(trips_text)

        for trip in trip_list:
            trip_record = {
                "traveler_id": traveler["traveler_id"],
                "age": traveler["age"],
                "occupation": traveler["occupation"],
                "income": traveler["income"],
                "flexibility": traveler["flexible"],
                **trip
            }
            all_trips.append(trip_record)

    trip_df = pd.DataFrame(all_trips)

    print("\nParsed trip records:")
    trip_df.to_csv("traveler_trip_types.csv", index = False)
else:
    trip_df = pd.read_csv("traveler_trip_types.csv")

def process_trip(trip_record, flexibilities, dt=10, max_shift=30):
    """Call API and pivot results for one traveler."""
    results = generate_decisions(trip_record, flexibilities=flexibilities, dt=dt, max_shift=max_shift)
    flexibility_pickup_earlier = trip_record["flexibility_pickup_earlier"]
    flexibility_dropoff_later = trip_record["flexibility_dropoff_later"]

    all_rows = []
    for d in results:
        row = trip_record.copy()
        pickup_shift_min = d.get("pickup_shift_min")
        dropoff_shift_min = d.get("dropoff_shift_min")
        decision = d.get("decision")

        if pickup_shift_min > flexibility_pickup_earlier or dropoff_shift_min > flexibility_dropoff_later:
            decision = "reject"

        row.update({
            "flexibility_type": d.get("flexibility_type"),
            "pickup_shift_min": pickup_shift_min,
            "dropoff_shift_min": dropoff_shift_min,
            "decision": decision,
        })
        all_rows.append(row)

    # Pivot the DataFrame
    decisions_df = pd.DataFrame(all_rows)
    decisions_df = decisions_df.pivot_table(
        index=[
            "traveler_id", "age", "occupation", "income", "flexibility",
            "trip_purpose", "departure_location", "arrival_location",
            "origin_gps", "destination_gps", "trip_distance_miles",
            "trip_duration_minutes", "departure_time_window", "arrival_time_window",
            "flexibility_pickup_earlier", "flexibility_dropoff_later",
            "pickup_shift_min", "dropoff_shift_min"
        ],
        columns="flexibility_type",
        values="decision",
        aggfunc="first"
    ).reset_index()

    decisions_df.columns.name = None
    return decisions_df

if REGENERATE_DECISIONS:
    # ===== Parallel Processing with Progress Bar =====
    output_file = "traveler_decisions_augmented.csv"
    # Remove old file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    header_written = os.path.exists(output_file)
    max_workers = 8  # adjust for rate limits / system

    # Pre-convert dataframe rows to dicts to avoid locking overhead
    trip_records = [trip_df.iloc[i].to_dict() for i in range(trip_df.shape[0])]
    flexibilities = list(trip_df["flexibility"].unique())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_trip, record, flexibilities)
            for record in trip_records
        ]
        with tqdm(total=len(futures), desc="Processing trips") as pbar:
            for future in as_completed(futures):
                try:
                    decisions_df = future.result()
                    header = not header_written
                    decisions_df.to_csv(output_file, mode='a', header=header, index=False)
                    header_written = True
                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    pbar.update(1)  # ✅ update progress bar as each finishes
decisions_df = pd.read_csv(output_file)
