import pandas as pd
from pandas.testing import assert_series_equal
import os
from unittest import TestCase
from src.calc_cell_counts import OGU_ID_KEY, OGU_READ_COUNT_KEY, \
    OGU_LEN_IN_BP_KEY, OGU_GDNA_MASS_NG_KEY, \
    OGU_CELLS_PER_G_OF_GDNA_KEY, \
    _calc_ogu_gdna_mass_ng_series_for_sample, \
    _calc_ogu_genomes_per_g_of_gdna_series_for_sample


class TestCalcCellCounts(TestCase):
    def setUp(self):
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def test__calc_ogu_gdna_mass_ng_series_for_sample(self):
        # Inputs and expected results are taken from cell [5] of the
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1, with an
        # added column holding the "reads2weigth" [sic] values (which are
        # OGU_GDNA_MASS_NG_KEY values). The "GenomeLenght" [sic] column is
        # the OGU_LEN_IN_BP_KEY, the "Species" column is OGU_ID_KEY, and the
        # RawCounts column is OGU_READ_COUNT_KEY.
        input_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            OGU_READ_COUNT_KEY: [79950, 93024, 86188, 45441, 31185, 24929,
                                 1975, 26130, 22303, 19783, 14478, 12145,
                                 14609],
            OGU_LEN_IN_BP_KEY: [1904788.333, 4373730, 5033120.886, 3861016,
                                2453028, 2031251, 89264, 2292986, 2204851,
                                2484878.333, 2058778.25, 1680673.6, 2116567],

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

        slope = 1.24487652379132
        intercept = -6.77539505390338

        input_df = pd.DataFrame(input_dict)
        expected_series = pd.Series(expected_dict[OGU_GDNA_MASS_NG_KEY],
                                    index=expected_dict[OGU_ID_KEY],
                                    name=OGU_GDNA_MASS_NG_KEY)
        expected_series.index.name = OGU_ID_KEY

        output_series = _calc_ogu_gdna_mass_ng_series_for_sample(
            input_df, slope, intercept)

        assert_series_equal(expected_series, output_series)

    def test__calc_ogu_genomes_per_g_of_gdna_series_for_sample(self):
        # Inputs and expected results are taken from cell [5] of the
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1, with an
        # added column holding the "reads2weigth" [sic] values (which are
        # OGU_GDNA_MASS_NG_KEY values). The "GenomeLenght" [sic] column is
        # the OGU_LEN_IN_BP_KEY, the "Species" column is OGU_ID_KEY, and the
        # CellNumber column is OGU_CELLS_PER_G_OF_GDNA_KEY.
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

        # NOTE: the is_test flag is set to True to use the *truncated* value of
        # Avogadro's number that is used in the notebook that is the source of
        # the expected results.  It should never be set to True in production.
        output_series = _calc_ogu_genomes_per_g_of_gdna_series_for_sample(
            input_df, is_test=True)

        assert_series_equal(expected_series, output_series)

