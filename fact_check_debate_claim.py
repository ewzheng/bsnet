import os
import re
import sys
import requests
from difflib import SequenceMatcher


def detect_claim_type(statement: str) -> str:
    s = statement.lower().strip()

    def has_any(*patterns: str) -> bool:
        return any(re.search(pattern, s, flags=re.IGNORECASE) for pattern in patterns)

    # 1. INFLATION / PRICES / COST OF LIVING
    if has_any(
        r"\binflation\b",
        r"\bcpi\b",
        r"\bconsumer price index\b",
        r"\bprices?\b",
        r"\bcost of living\b",
        r"\bgrocery prices?\b",
        r"\bfood prices?\b",
        r"\bgas prices?\b",
        r"\baffordability\b",
        r"\bheadline inflation\b",
        r"\bcore inflation\b",
        r"\bpce\b",
        r"\bpersonal consumption expenditures\b",
        r"\bdeflation\b",
        r"\bdisinflation\b",
    ):
        return "inflation_prices"

    # 2. JOBS / UNEMPLOYMENT / WAGES
    if has_any(
        r"\bunemployment\b",
        r"\bunemployment rate\b",
        r"\bjobless\b",
        r"\bjobless rate\b",
        r"\bjobs?\b",
        r"\bjob creation\b",
        r"\bjobs added\b",
        r"\bpayrolls?\b",
        r"\bnonfarm payrolls?\b",
        r"\blabor market\b",
        r"\blabour market\b",
        r"\blabor force\b",
        r"\blabour force\b",
        r"\bwages?\b",
        r"\bearnings?\b",
        r"\bmedian wage\b",
        r"\bworkers?\b",
    ):
        return "jobs_unemployment_wages"

    # 3. GDP / GROWTH / RECESSION
    if has_any(
        r"\bgdp\b",
        r"\bgross domestic product\b",
        r"\beconomic growth\b",
        r"\bgrowth rate\b",
        r"\bthe economy grew\b",
        r"\bthe economy shrank\b",
        r"\brecession\b",
        r"\bcontraction\b",
        r"\bexpansion\b",
        r"\bquarterly growth\b",
        r"\bannual growth\b",
        r"\boutput\b",
        r"\bproductivity\b",
    ):
        return "gdp_growth_recession"

    # 4. IMMIGRATION / BORDER / ASYLUM
    if has_any(
        r"\bimmigration\b",
        r"\bimmigrants?\b",
        r"\bmigrants?\b",
        r"\bborder\b",
        r"\bsouthern border\b",
        r"\billegal immigration\b",
        r"\bundocumented\b",
        r"\basylum\b",
        r"\brefugees?\b",
        r"\bdeport(?:ation|ed|ing)?\b",
        r"\bborder crossings?\b",
        r"\bencounters?\b",
        r"\bcatch and release\b",
        r"\bvisa overstay\b",
        r"\bwall\b",
        r"\bremain in mexico\b",
    ):
        return "immigration_border"

    # 5. CRIME / PUBLIC SAFETY
    if has_any(
        r"\bcrime\b",
        r"\bviolent crime\b",
        r"\bproperty crime\b",
        r"\bmurder\b",
        r"\bhomicide\b",
        r"\bassault\b",
        r"\brobbery\b",
        r"\bshootings?\b",
        r"\bgun violence\b",
        r"\bfentanyl\b",
        r"\boverdose\b",
        r"\bdrug trafficking\b",
        r"\bcarjacking\b",
        r"\bpolice\b",
        r"\blaw enforcement\b",
        r"\bpublic safety\b",
    ):
        return "crime_public_safety"

    # 6. TAXES / SPENDING / DEFICIT / DEBT
    if has_any(
        r"\btax(?:es|ed|ing)?\b",
        r"\btax cut\b",
        r"\btax increase\b",
        r"\bcorporate tax\b",
        r"\bincome tax\b",
        r"\bpayroll tax\b",
        r"\btariffs?\b",
        r"\bgovernment spending\b",
        r"\bfederal spending\b",
        r"\bdeficit\b",
        r"\bbudget deficit\b",
        r"\bnational debt\b",
        r"\bdebt\b",
        r"\bdebt to gdp\b",
        r"\bappropriations?\b",
        r"\bsubsid(?:y|ies)\b",
        r"\bspending bill\b",
    ):
        return "taxes_spending_deficit_debt"

    # 7. HEALTHCARE / INSURANCE / DRUG PRICES
    if has_any(
        r"\bhealth care\b",
        r"\bhealthcare\b",
        r"\binsurance\b",
        r"\bhealth insurance\b",
        r"\bmedicare\b",
        r"\bmedicaid\b",
        r"\baca\b",
        r"\baffordable care act\b",
        r"\bobamacare\b",
        r"\bdrug prices?\b",
        r"\bprescription drugs?\b",
        r"\bhospital costs?\b",
        r"\bmedical bills?\b",
        r"\bcoverage\b",
        r"\buninsured\b",
        r"\bpublic option\b",
    ):
        return "healthcare"

    # 8. ABORTION / REPRODUCTIVE RIGHTS
    if has_any(
        r"\babortion\b",
        r"\breproductive rights?\b",
        r"\breproductive health\b",
        r"\broe\b",
        r"\broe v\.? wade\b",
        r"\bdobbs\b",
        r"\bivf\b",
        r"\bconception\b",
        r"\bfetal\b",
        r"\bheartbeat bill\b",
        r"\b15-week ban\b",
        r"\b6-week ban\b",
        r"\bplanned parenthood\b",
        r"\bmaternal health\b",
    ):
        return "abortion_reproductive_policy"

    # 9. ENERGY / CLIMATE
    if has_any(
        r"\benergy\b",
        r"\boil\b",
        r"\bcrude\b",
        r"\bnatural gas\b",
        r"\bgasoline\b",
        r"\bgas prices?\b",
        r"\bdrilling\b",
        r"\bfracking\b",
        r"\bpipeline\b",
        r"\bstrategic petroleum reserve\b",
        r"\bcoal\b",
        r"\bnuclear\b",
        r"\bsolar\b",
        r"\bwind\b",
        r"\belectric grid\b",
        r"\belectricity prices?\b",
        r"\bclimate\b",
        r"\bclimate change\b",
        r"\bglobal warming\b",
        r"\bcarbon\b",
        r"\bemissions\b",
        r"\bgreenhouse gas\b",
    ):
        return "energy_climate"

    # 10. ELECTIONS / VOTING / OFFICEHOLDERS
    if has_any(
        r"\belection\b",
        r"\belections\b",
        r"\bvoting\b",
        r"\bvote\b",
        r"\bballot\b",
        r"\bmail[- ]in voting\b",
        r"\bearly voting\b",
        r"\bvoter fraud\b",
        r"\belection fraud\b",
        r"\brigged\b",
        r"\bcertif(?:y|ication)\b",
        r"\belectoral college\b",
        r"\bpopular vote\b",
        r"\bturnout\b",
        r"\bpresident\b",
        r"\bvice president\b",
        r"\bgovernor\b",
        r"\bsenator\b",
        r"\bmember of congress\b",
        r"\bspeaker\b",
        r"\bsecretary of state\b",
        r"\battorney general\b",
    ):
        return "elections_voting_officeholders"

    # 11. EDUCATION
    if has_any(
        r"\beducation\b",
        r"\bschools?\b",
        r"\bpublic schools?\b",
        r"\bprivate schools?\b",
        r"\bteachers?\b",
        r"\bstudents?\b",
        r"\btest scores?\b",
        r"\bliteracy\b",
        r"\breading scores?\b",
        r"\bmath scores?\b",
        r"\bcollege\b",
        r"\btuition\b",
        r"\bstudent loans?\b",
        r"\bschool choice\b",
        r"\bcharter schools?\b",
        r"\bcurriculum\b",
    ):
        return "education"

    if has_any(
        r"\b\d+(?:\.\d+)?\s*%\b",
        r"\bpercent\b",
        r"\bpercentage points?\b",
        r"\brate\b",
        r"\bmillions?\b",
        r"\bbillions?\b",
        r"\btrillions?\b",
    ):
        return "numeric_policy_claim"

    return "generic"


def google_fact_check_search(query: str, api_key: str, language_code: str = "en-US", page_size: int = 10) -> dict:
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query,
        "key": api_key,
        "languageCode": language_code,
        "pageSize": page_size,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def pick_best_review(claim_text: str, api_response: dict) -> dict | None:
    claims = api_response.get("claims", [])
    best = None
    best_score = -1.0

    for claim in claims:
        returned_claim_text = claim.get("text", "")
        score = similarity(claim_text, returned_claim_text)

        for review in claim.get("claimReview", []):
            combined = {
                "matched_claim_text": returned_claim_text,
                "claimant": claim.get("claimant"),
                "claim_date": claim.get("claimDate"),
                "publisher_name": review.get("publisher", {}).get("name"),
                "publisher_site": review.get("publisher", {}).get("site"),
                "textual_rating": review.get("textualRating"),
                "review_title": review.get("title"),
                "review_date": review.get("reviewDate"),
                "url": review.get("url"),
                "score": score,
            }
            if score > best_score:
                best = combined
                best_score = score

    return best


def main() -> None:
    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if not api_key:
        print("Error: set the GOOGLE_FACTCHECK_API_KEY environment variable first.")
        sys.exit(1)

    if len(sys.argv) > 1:
        claim = " ".join(sys.argv[1:]).strip()
    else:
        claim = input("Enter a claim: ").strip()

    if not claim:
        print("Error: claim cannot be empty.")
        sys.exit(1)

    claim_type = detect_claim_type(claim)
    print(f"Detected claim type: {claim_type}")

    try:
        data = google_fact_check_search(claim, api_key=api_key)
    except requests.HTTPError as e:
        print(f"HTTP error calling Google Fact Check API: {e}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Network error calling Google Fact Check API: {e}")
        sys.exit(1)

    best = pick_best_review(claim, data)

    if not best:
        print("No fact-check results found.")
        return

    print("\nBest matching fact-check result")
    print("-" * 40)
    print(f"Claim entered:      {claim}")
    print(f"Matched claim:      {best.get('matched_claim_text')}")
    print(f"Truth rating:       {best.get('textual_rating') or 'N/A'}")
    print(f"Source:             {best.get('publisher_name') or 'N/A'}")
    print(f"Source site:        {best.get('publisher_site') or 'N/A'}")
    print(f"Review title:       {best.get('review_title') or 'N/A'}")
    print(f"Review date:        {best.get('review_date') or 'N/A'}")
    print(f"Review URL:         {best.get('url') or 'N/A'}")
    print(f"Claimant:           {best.get('claimant') or 'N/A'}")
    print(f"Match score:        {best.get('score'):.3f}")


if __name__ == "__main__":
    main()
