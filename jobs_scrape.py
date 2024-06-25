import csv
from jobspy import scrape_jobs

def scrape_my_jobs(location, position, results_wanted):
    jobs = scrape_jobs(
        site_name=["linkedin"],
        search_term=position,
        location=location,
        results_wanted=results_wanted,
        linkedin_fetch_description=True  # get full description and direct job url for LinkedIn (slower)
    )
    # Ensure jobs is a list of dictionaries with a 'description' key
    return jobs
