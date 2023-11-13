import biom.table
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import os
from unittest import TestCase
from src.calc_cell_counts import OGU_ID_KEY, OGU_LEN_IN_BP_KEY, \
    OGU_GDNA_MASS_NG_KEY, OGU_CELLS_PER_G_OF_GDNA_KEY, \
    _calc_ogu_genomes_per_g_of_gdna_series_for_sample


class TestCalcCellCounts(TestCase):
    def setUp(self):
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def test__calc_ogu_genomes_per_g_of_gdna_series_for_sample(self):
        input_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                       "Escherichia coli", "Tyzzerella nexilis",
                       "Prevotella sp. oral taxon 299",
                       "Streptococcus mitis", "Leptolyngbya valderiana",
                       "Neisseria subflava", "Neisseria flavescens",
                       "Fusobacterium periodonticum",
                       "Streptococcus pneumoniae", "Haemophilus influenzae",
                       "Veillonella dispar"],
            OGU_LEN_IN_BP_KEY: [1904788.333, 4373730, 5033120.886, 3861016,
                                2453028, 2031251, 89264, 2292986, 2204851,
                                2484878.333, 2058778.25, 1680673.6, 2116567],
            OGU_GDNA_MASS_NG_KEY: [0.54168919427718800000,
                                   0.65408448128256300000,
                                   0.59479652205780900000,
                                   0.26809831701850500000,
                                   0.16778538250235500000,
                                   0.12697002941967100000,
                                   0.00540655563393439000,
                                   0.13462933928829500000,
                                   0.11054062877622500000,
                                   0.09521377825242420000,
                                   0.06455278517415270000,
                                   0.05187010729728740000,
                                   0.06528070535624650000]
        }

        expected_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            OGU_CELLS_PER_G_OF_GDNA_KEY: [263469.8016, 138550.8742,
                                          109485.9657, 64330.93757,
                                          63369.31482, 57911.52782,
                                          56114.06446, 54395.84228,
                                          46448.32735, 35499.48595,
                                          29049.10845, 28593.0947,
                                          28574.60346]
        }

        input_df = pd.DataFrame(input_dict)
        expected_series = pd.Series(expected_dict[OGU_CELLS_PER_G_OF_GDNA_KEY],
                                    index=expected_dict[OGU_ID_KEY])
        expected_series.index.name = OGU_ID_KEY

        output_series = _calc_ogu_genomes_per_g_of_gdna_series_for_sample(
            input_df, is_test=True)

        assert_series_equal(expected_series, output_series)
