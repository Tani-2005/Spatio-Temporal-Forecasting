import json
import random
import geonamescache
import os

def generate_global_database():
    print("🌍 Initializing Global Node Generation...")
    
    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()
    cities = gc.get_cities()

    # 1. Group all cities by their country code
    cities_by_country = {}
    for city_id, city_data in cities.items():
        cc = city_data['countrycode']
        if cc not in cities_by_country:
            cities_by_country[cc] = []
        cities_by_country[cc].append(city_data)

    world_database = {}

    # 2. Iterate through all 195+ countries
    for cc, country_data in countries.items():
        country_name = country_data['name']
        
        # Get all cities for this specific country
        country_cities = cities_by_country.get(cc, [])
        if not country_cities:
            continue
            
        # 3. Sort cities by population to get the Top 5 Urban Centers
        country_cities = sorted(country_cities, key=lambda x: x.get('population', 0), reverse=True)
        top_5 = country_cities[:5]
        
        # We use the largest city's coordinates as the geographic center for the country map
        country_lat = top_5[0]['latitude']
        country_lon = top_5[0]['longitude']
        
        # 4. Generate realistic baseline risk metrics
        base_risk = random.randint(40, 180)
        
        city_list = []
        for c in top_5:
            city_list.append({
                "name": c['name'],
                "lat": c['latitude'],
                "lon": c['longitude'],
                "base_risk": base_risk + random.randint(-20, 40) # Cities fluctuate around national baseline
            })
            
        # 5. Build the final data node for this country
        world_database[country_name] = {
            "lat": country_lat,
            "lon": country_lon,
            "geo": cc, # Google Trends 2-letter country code!
            "base_risk": base_risk,
            "vuln_score": random.randint(20, 90),
            "capacity": random.randint(100, 500),
            "cities": city_list
        }

    # Ensure the data folder exists
    os.makedirs("data", exist_ok=True)

    # 6. Save out the massive JSON file
    with open("data/global_nodes.json", "w", encoding="utf-8") as f:
        json.dump(world_database, f, indent=4)
        
    print(f"✅ Success! Generated data for {len(world_database)} countries and saved to data/global_nodes.json")

if __name__ == "__main__":
    generate_global_database()