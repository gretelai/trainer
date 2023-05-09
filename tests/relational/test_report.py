from datetime import datetime

from lxml import html

from gretel_trainer.relational.core import Scope
from gretel_trainer.relational.report.report import ReportPresenter, ReportRenderer
from gretel_trainer.relational.table_evaluation import TableEvaluation


def _evals_from_rel_data(rel_data):
    d = {
        "synthetic_data_quality_score": {"score": 90, "grade": "Excellent"},
        "privacy_protection_level": {"score": 2, "grade": "Good"},
    }
    evals = {}
    for table in rel_data.list_all_tables(Scope.PUBLIC):
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

    # DEV ONLY if you want to save a local copy to look at
    # with open("report.html", 'w') as f:
    #     f.write(html_content)

    tree = html.fromstring(html_content)

    # Top level scores
    assert (
        len(
            tree.xpath(
                '//div[contains(@class, "test-report-main-score")]'
                + '//div[contains(@class, "score-container")]'
            )
        )
        == 2
    )
    # SQS score label and bottom text
    assert (
        tree.xpath(
            '//div[contains(@class, "test-report-main-score")]'
            + '//div[contains(@class, "score-container")]'
            + '//span[contains(@class, "label")]'
        )[0].text.strip()
        == "Excellent"
    )
    assert (
        tree.xpath(
            '//div[contains(@class, "test-report-main-score")]'
            + '//div[contains(@class, "score-container")]'
            + '//span[contains(@class, "score-container-text")]'
        )[0].text
        == "Composite"  # <br /> cuts off the rest
    )
    # PPL score label and bottom text
    assert (
        tree.xpath(
            '//div[contains(@class, "test-report-main-score")]'
            + '//div[contains(@class, "score-container")]'
            + '//span[contains(@class, "label")]'
        )[1].text.strip()
        == "Good"
    )
    assert (
        tree.xpath(
            '//div[contains(@class, "test-report-main-score")]'
            + '//div[contains(@class, "score-container")]'
            + '//span[contains(@class, "score-container-text")]'
        )[0].text.strip()
        == "Composite"  # <br /> cuts off the rest
    )

    # Table relationships
    assert (
        len(
            tree.xpath(
                '//section[contains(@class, "test-table-relationships")]' + "//tr"
            )
        )
        == 7  # Header plus six tables
    )
    relations_data_rows = tree.xpath(
        '//section[contains(@class, "test-table-relationships")]' + "//tr"
    )[1:]
    # First row, Table name td, bold tag wrapping table name
    assert (
        relations_data_rows[0].getchildren()[0].getchildren()[0].text
        == "distribution_center"
    )
    # pk column/td, each is a span, unpack then text
    pks = [row.getchildren()[1].getchildren()[0].text for row in relations_data_rows]
    for pk in pks:
        assert pk == "id"
    # First row has no fk's
    assert len(relations_data_rows[0].getchildren()[2].getchildren()) == 0
    # Third row has two fk's
    assert len(relations_data_rows[2].getchildren()[2].getchildren()) == 2

    # SQS score table
    assert (
        len(tree.xpath('//section[contains(@class, "test-sqs-results")]' + "//tr"))
        == 7  # Header plus six tables again
    )
    assert (
        len(
            tree.xpath(
                '//section[contains(@class, "test-sqs-results")]'
                + "//tr"
                + '//span[contains(@class, "sqs-table-score")]'
            )
        )
        == 12  # Six tables, each has two numeric scores
    )
    assert (
        len(
            tree.xpath(
                '//section[contains(@class, "test-sqs-results")]'
                + "//tr"
                + '//span[contains(@class, "label")]'
            )
        )
        == 12  # Six tables, each has two grade labels
    )
    assert (
        len(
            tree.xpath(
                '//section[contains(@class, "test-sqs-results")]'
                + "//tr"
                + '//span[contains(@class, "sqs-table-link")]'
            )
        )
        == 12  # Six tables, each has two linked reports
    )
    # Check the first report link
    assert (
        tree.xpath(
            '//section[contains(@class, "test-sqs-results")]'
            + "//tr"
            + '//span[contains(@class, "sqs-table-link")]'
            + "/a/@href"
        )[0]
        == "synthetics_individual_evaluation_distribution_center.html"
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

    # DEV ONLY if you want to save a local copy to look at
    # with open("report.html", 'w') as f:
    #     f.write(html_content)

    tree = html.fromstring(html_content)

    # Two scores at top
    assert (
        len(
            tree.xpath(
                '//div[contains(@class, "test-report-main-score")]'
                + '//div[contains(@class, "score-container")]'
            )
        )
        == 2
    )

    # Table relationships
    assert (
        len(
            tree.xpath(
                '//section[contains(@class, "test-table-relationships")]' + "//tr"
            )
        )
        == 4  # Header plus three tables
    )

    # SQS score table
    assert (
        len(tree.xpath('//section[contains(@class, "test-sqs-results")]' + "//tr"))
        == 4  # Header plus three tables again
    )
    assert (
        len(
            tree.xpath(
                '//section[contains(@class, "test-sqs-results")]'
                + "//tr"
                + '//span[contains(@class, "sqs-table-score")]'
            )
        )
        == 6  # Three tables, each has two numeric scores
    )
    assert (
        len(
            tree.xpath(
                '//section[contains(@class, "test-sqs-results")]'
                + "//tr"
                + '//span[contains(@class, "label")]'
            )
        )
        == 6  # Three tables, each has two grade labels
    )
    assert (
        len(
            tree.xpath(
                '//section[contains(@class, "test-sqs-results")]'
                + "//tr"
                + '//span[contains(@class, "sqs-table-link")]'
            )
        )
        == 6  # Three tables, each has two linked reports
    )
    # Check the first report link
    assert (
        tree.xpath(
            '//section[contains(@class, "test-sqs-results")]'
            + "//tr"
            + '//span[contains(@class, "sqs-table-link")]'
            + "/a/@href"
        )[0]
        == "synthetics_individual_evaluation_atom.html"
    )


def test_source_data_including_json(documents):
    # Fake these
    evaluations = _evals_from_rel_data(documents)

    presenter = ReportPresenter(
        rel_data=documents,
        evaluations=evaluations,
        now=datetime.utcnow(),
        run_identifier="run_identifier",
    )

    html_content = ReportRenderer().render(presenter)

    # DEV ONLY if you want to save a local copy to look at
    # with open("report.html", 'w') as f:
    #     f.write(html_content)

    tree = html.fromstring(html_content)

    relations_data_rows = tree.xpath(
        '//section[contains(@class, "test-table-relationships")]' + "//tr"
    )[1:]

    # Ensure public names, not invented table names, are displayed
    table_names = [
        # Row, Table name td, bold tag wrapping table name
        row.getchildren()[0].getchildren()[0].text
        for row in relations_data_rows
    ]
    assert table_names == ["payments", "purchases", "users"]
