"""Capture documentation screenshots after Streamlit reports each view ready."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as conditions
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "docs" / "images"
PAGES = {
    "Overview": ("01-overview.png", "Autonomous ML Experimenter", 1100),
    "Research Journey": ("02-research-journey.png", "Research Journey", 1100),
    "Experiment Comparison": (
        "03-experiment-lineage.png",
        "Experiment Comparison",
        1250,
    ),
    "Monitoring": ("04-monitoring.png", "Monitoring", 1200),
}


def capture(page: str) -> None:
    filename, heading, height = PAGES[page]
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--hide-scrollbars")
    options.add_argument(f"--window-size=1440,{height}")
    options.page_load_strategy = "eager"
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    wait = WebDriverWait(driver, 40)
    try:
        driver.get(f"http://localhost:8501/?page={quote(page)}")
        wait.until(
            conditions.text_to_be_present_in_element(
                (By.CSS_SELECTOR, "[data-testid='stAppViewContainer']"), heading
            )
        )
        wait.until(
            lambda browser: (
                len(
                    browser.find_elements(
                        By.CSS_SELECTOR, "[data-testid='stSidebar'] [role='radiogroup'] label"
                    )
                )
                >= 6
            )
        )
        if page == "Overview":
            wait.until(
                lambda browser: any(
                    element.text.strip()
                    for element in browser.find_elements(
                        By.CSS_SELECTOR, "[data-testid='stMetricValue']"
                    )
                )
            )
        else:
            wait.until(conditions.presence_of_element_located((By.CSS_SELECTOR, ".js-plotly-plot")))
        wait.until(
            lambda browser: not browser.find_elements(By.CSS_SELECTOR, "[data-testid='stSkeleton']")
        )
        time.sleep(1)
        target = OUTPUT / filename
        if not driver.save_screenshot(str(target)):
            raise RuntimeError(f"failed to capture {filename}")
        print(f"captured {filename}")
    finally:
        driver.quit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", choices=PAGES)
    arguments = parser.parse_args()
    for page in [arguments.page] if arguments.page else PAGES:
        capture(page)


if __name__ == "__main__":
    main()
