"""Focused deterministic grounding regressions; no model API use."""
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from target_plan import dimension_evidence_is_grounded


CORE_DIMENSIONS = ("actor_subject", "predicate_object", "scope_location")
VIRGIN_TARGET = (
    "Virgin Galactic will commence commercial service during the second "
    "quarter of 2023."
)
VIRGIN_HEADING = (
    "Virgin Galactic 2023 Form 10-K | SEC filing dated February 27, 2024"
)
VIRGIN_BODY = (
    "In June 2023, 'Galactic 01,' marked the start of commercial service."
)
BOEING_TARGET = (
    "Boeing will complete the Starliner crewed flight test by December 31, "
    "2023."
)
BOEING_CONTEXT = (
    "Boeing Q3 2024 Form 10-Q | SEC filing dated October 23, 2024\n"
    "National Aeronautics and Space Administration has contracted us to "
    "design and build the CST-100 Starliner spacecraft to transport crews "
    "to the International Space Station (ISS). During 2023, we increased "
    "the reach-forward loss by $288 primarily as a result of delaying the "
    "Crewed Flight Test (CFT) following notification by a parachute supplier "
    "of an issue identified through testing.\nThe CFT launched on June 5, "
    "2024, and docked with the ISS."
)


class TargetPlanGroundingTests(unittest.TestCase):
    def assert_core_rejected(self, probe, basis):
        for dimension in CORE_DIMENSIONS:
            with self.subTest(dimension=dimension, probe=probe, basis=basis):
                self.assertFalse(dimension_evidence_is_grounded(
                    dimension, probe, basis, "supported"))

    def test_contiguous_virgin_excerpt_grounds_all_dimensions(self):
        basis = VIRGIN_HEADING + "\n" + VIRGIN_BODY

        for dimension in (*CORE_DIMENSIONS, "time"):
            with self.subTest(dimension=dimension):
                self.assertTrue(dimension_evidence_is_grounded(
                    dimension, VIRGIN_TARGET, basis, "supported"))

    def test_split_virgin_heading_and_body_do_not_ground_core(self):
        # The verifier still needs one contiguous span that binds issuer,
        # action and object; it must not assemble that binding across spans.
        self.assert_core_rejected(VIRGIN_TARGET, VIRGIN_HEADING)
        self.assert_core_rejected(VIRGIN_TARGET, VIRGIN_BODY)

    def test_september_does_not_ground_second_quarter(self):
        basis = VIRGIN_HEADING + "\n" + (
            "In September 2023, 'Galactic 01,' marked the start of commercial "
            "service."
        )

        self.assertTrue(dimension_evidence_is_grounded(
            "predicate_object", VIRGIN_TARGET, basis, "supported"))
        self.assertFalse(dimension_evidence_is_grounded(
            "time", VIRGIN_TARGET, basis, "supported"))

    def test_end_is_not_equivalent_to_commence(self):
        basis = VIRGIN_HEADING + "\n" + (
            "In June 2023, 'Galactic 01,' marked the end of commercial service."
        )
        self.assert_core_rejected(VIRGIN_TARGET, basis)

    def test_closed_start_equivalence_includes_morphology(self):
        forms = (
            "commence", "commenced", "commences", "commencing",
            "begin", "began", "begun", "begins", "beginning",
            "start", "started", "starts", "starting",
        )
        for form in forms:
            with self.subTest(form=form):
                self.assertTrue(dimension_evidence_is_grounded(
                    "actor_subject",
                    f"Virgin Galactic {form} commercial service.",
                    "Virgin Galactic started commercial service.",
                    "supported",
                ))

    def test_wrong_actor_object_and_predicate_remain_rejected(self):
        for probe, basis in (
            ("Officials denied the report.", "Officials confirmed the report."),
            ("Alice bought Widget.", "Bob bought Widget."),
            ("Alice bought Widget.", "Alice bought Gadget."),
            ("Alice will buy Widget.", "Alice sold Widget."),
        ):
            self.assert_core_rejected(probe, basis)

    def test_role_order_and_conjunction_scope_remain_rejected(self):
        for probe, basis in (
            ("Alice defeated Bob.", "Bob defeated Alice."),
            (
                "The agency alleged that Alice failed and Bob cheated.",
                "The agency alleged that Alice failed. Bob cheated.",
            ),
            ("苹果公司在北京收购微软。", "苹果公司在上海发布新品。"),
        ):
            self.assert_core_rejected(probe, basis)

    def test_contradiction_requires_ordered_event_identity(self):
        header = "Boeing Q3 2024 Form 10-Q | SEC filing dated October 23, 2024"
        acronym_only = "The CFT launched on June 5, 2024, and docked with the ISS."

        self.assertFalse(dimension_evidence_is_grounded(
            "time", BOEING_TARGET, header, "contradicted"))
        self.assertFalse(dimension_evidence_is_grounded(
            "time", BOEING_TARGET, acronym_only, "contradicted"))
        self.assertTrue(dimension_evidence_is_grounded(
            "time", BOEING_TARGET, BOEING_CONTEXT, "contradicted"))

    def test_simple_object_identity_still_allows_a_grounded_contradiction(self):
        self.assertTrue(dimension_evidence_is_grounded(
            "time", "The bridge reopened before Friday.",
            "The bridge remains closed until Friday.", "contradicted"))


if __name__ == "__main__":
    unittest.main()
