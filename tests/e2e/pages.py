"""Page Object for the CarePilot Streamlit app.

Locators use roles, labels and Streamlit's own data-testid attributes, never CSS classes
(Streamlit's generated class names change between versions).
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

# First run loads embedding + reranker models; explanations run several retrievals.
MODEL_TIMEOUT_MS = 180_000


class CarePilotPage:
    def __init__(self, page: Page, base_url: str):
        self.page = page
        self.base_url = base_url
        self.sidebar = page.get_by_test_id("stSidebar")
        self.main = page.get_by_test_id("stMainBlockContainer")

    def goto(self) -> None:
        self.page.goto(self.base_url)
        expect(self.page.get_by_role("heading", name="CarePilot").first).to_be_visible(
            timeout=60_000
        )
        expect(self.sidebar.get_by_text("Using:")).to_be_visible()  # first script run done
        self.wait_idle()

    def tab(self, name: str) -> None:
        self.page.get_by_role("tab", name=name).click()

    def wait_idle(self) -> None:
        """Wait until Streamlit finished rerunning the script.

        Every widget change triggers a rerun; acting before it ends races the old UI
        (for example submitting the previous preset's values).
        """
        expect(self.page.get_by_test_id("stStatusWidget")).to_have_count(0, timeout=60_000)
        # During a rerun the old widgets stay on screen marked stale next to the new ones.
        expect(self.page.locator("[data-stale='true']")).to_have_count(0, timeout=60_000)

    def _pick(self, select: Locator, option: str, attempts: int = 3) -> None:
        """Open the dropdown and choose an option.

        A click that lands while Streamlit re-renders can leave the list closed, so the open
        is retried, each time waiting for the option itself rather than sleeping.
        """
        item = self.page.get_by_role("option", name=option, exact=True)
        for attempt in range(attempts):
            self.wait_idle()
            select.click()
            try:
                item.click(timeout=5_000)  # waits for visible + stable
                break
            except PlaywrightTimeoutError:
                if attempt == attempts - 1:
                    raise
                self.page.keyboard.press("Escape")  # close a half-open list before retrying
        self.wait_idle()

    # ----- risk tab -----
    def choose_condition(self, label: str) -> None:
        self.main.get_by_text(label, exact=True).click()
        self.wait_idle()

    def choose_preset(self, name: str) -> None:
        self._pick(self.main.get_by_test_id("stSelectbox").first, name)

    def set_number(self, label: str, value: float) -> None:
        self.wait_idle()
        box = self.main.get_by_label(label, exact=True)
        box.fill(str(value))
        box.press("Tab")
        expect(box).to_have_value(str(value))

    def estimate(self) -> None:
        self.wait_idle()
        self.main.get_by_role("button", name="Estimate risk").click()

    @property
    def risk_metric(self) -> Locator:
        return self.main.get_by_test_id("stMetric").filter(has_text="Estimated risk")

    def band(self, name: str) -> Locator:
        return self.main.get_by_text(name, exact=True)

    @property
    def sources_header(self) -> Locator:
        return self.main.get_by_text("Sources", exact=True)

    @property
    def source_items(self) -> Locator:
        # numbered source expanders only (claim-check and X-ray panels are expanders too)
        return self.main.get_by_test_id("stExpander").filter(
            has_text=re.compile(r"\[\d+\] (NIDDK|NHLBI|World Health)")
        )

    # ----- ask tab -----
    def ask(self, question: str) -> None:
        self.main.get_by_label("Your question").fill(question)
        self.main.get_by_role("button", name="Ask", exact=True).click()

    def ask_example(self, question: str) -> None:
        self.main.get_by_role("button", name=question).click()

    @property
    def emergency_banner(self) -> Locator:
        return self.main.get_by_test_id("stAlertContentError")

    @property
    def not_found(self) -> Locator:
        return self.main.get_by_text("not available in the provided documents")
