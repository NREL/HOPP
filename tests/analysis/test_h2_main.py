import os
from pathlib import Path
import pandas as pd
from pytest import approx
from examples.H2_Analysis.h2_main import h2_main

class TestH2Main:
    def test_h2_main(self):
        """
        Test h2_main.py function
        """
        results_dir = Path(__file__).absolute().parent.parent.parent / "examples" / 'H2_Analysis' / 'results'
        expected_result_dir = Path(__file__).absolute().parent

        h2_main()
        df_produced = pd.read_csv(os.path.join(results_dir, 'H2_Analysis_Main.csv'))
        df_expected = pd.read_csv(os.path.join(expected_result_dir, 'Expected_H2_Analysis_Main.csv'))
        pd.testing.assert_frame_equal(df_produced, df_expected, check_exact=False, atol=10, check_dtype=False)