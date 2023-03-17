from datetime import datetime

import pytest
from lxml import html

from gretel_trainer.relational.core import (
    MultiTableException,
    RelationalData,
    TableEvaluation,
)
from gretel_trainer.relational.report.report import ReportPresenter, ReportRenderer


def _evals_from_rel_data(rel_data):
    d = {
        "synthetic_data_quality_score": {"score": 90, "grade": "Excellent"},
        "privacy_protection_level": {"score": 2, "grade": "Good"},
    }
    evals = {}
    for table in rel_data.list_all_tables():
        eval = TableEvaluation(cross_table_report_json=d, individual_report_json=d)
        evals[table] = eval
    return evals


def test_ecommerce_relational_data_report(ecom):
    # Fake these
    evaluations = _evals_from_rel_data(ecom)

    presenter = ReportPresenter(
        rel_data=ecom,
        evaluations=evaluations,
        now=datetime.utcnow(),
        run_identifier="run_identifier",
    )

    html_content = ReportRenderer().render(presenter)
    tree = html.fromstring(html_content)

    assert (
        len(
            tree.xpath(
                '//div[contains(@class, "test-report-main-score")]'
                + '//div[contains(@class, "score-container")]'
            )
        )
        == 2
    )


def test_mutagenesis_relational_data_report(mutagenesis):
    # Fake these
    evaluations = _evals_from_rel_data(mutagenesis)

    presenter = ReportPresenter(
        rel_data=mutagenesis,
        evaluations=evaluations,
        now=datetime.utcnow(),
        run_identifier="run_identifier",
    )

    html_content = ReportRenderer().render(presenter)
    tree = html.fromstring(html_content)

    assert (
        len(
            tree.xpath(
                '//div[contains(@class, "test-report-main-score")]'
                + '//div[contains(@class, "score-container")]'
            )
        )
        == 2
    )