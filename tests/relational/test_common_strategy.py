# import itertools
# import pytest
#
# import gretel_trainer.relational.strategies.common as common
#
#
# def test_get_frequencies(pets, trips):
#     pet_human_fk = pets.get_foreign_keys("pets")[0]
#     pet_fk_freqs = common.get_frequencies(pets, "pets", pet_human_fk)
#     assert pet_fk_freqs == [1, 1, 1, 1, 1]
#
#     trip_vehicle_type_fk = trips.get_foreign_keys("trips")[0]
#     trip_fk_freqs = common.get_frequencies(trips, "trips", trip_vehicle_type_fk)
#     assert sorted(trip_fk_freqs) == [5, 5, 30, 60]
#
#     some_fk = x.get_foreign_keys("x")[0]
#     fk_freqs = common.get_frequencies(x, "x", some_fk)
#     assert sorted(fk_freqs) == [0, 0, 1, 3]
#
#
# def test_get_values_with_frequencies_when_sizes_line_up_evenly():
#     values = ["a", "b", "c", "d"]
#     frequencies = [1, 1, 1, 1]
#     size = 4
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "b", "c", "d"]
#     assert actual == expected
#
#     values = ["a", "b", "c", "d", "e", "f", "g", "h"]
#     frequencies = [2, 2, 1, 1, 1, 1, 0, 0]
#     size = 8
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "a", "b", "b", "c", "d", "e", "f"]
#     assert actual == expected
#
#     values = ["a", "b", "c", "d", "e", "f", "g", "h"]
#     frequencies = [7, 1, 0, 0, 0, 0, 0, 0]
#     size = 8
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "a", "a", "a", "a", "a", "a", "b"]
#     assert actual == expected
#
#
# def test_get_values_with_frequencies_asking_for_more():
#     values = ["a", "b", "c", "d"]
#     frequencies = [1, 1, 1, 1]
#     size = 6
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "b", "c", "d", "a", "b"]
#     assert actual == expected
#
#     values = ["a", "b", "c", "d", "e", "f", "g", "h"]
#     frequencies = [2, 2, 1, 1, 1, 1, 0, 0]
#     size = 15
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "a", "b", "b", "c", "d", "e", "f", "a", "a", "b", "b", "c", "d", "e"]
#     assert actual == expected
#
#     values = ["a", "b", "c", "d", "e", "f", "g", "h"]
#     frequencies = [7, 1, 0, 0, 0, 0, 0, 0]
#     size = 17
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "a", "a", "a", "a", "a", "a", "b", "a", "a", "a", "a", "a", "a", "a", "b", "a"]
#     assert actual == expected
#
#
# def test_get_values_with_frequencies_asking_for_fewer():
#     values = ["a", "b", "c", "d"]
#     frequencies = [1, 1, 1, 1]
#     size = 3
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "b", "c"]
#     assert actual == expected
#
#     values = ["a", "b", "c", "d", "e", "f", "g", "h"]
#     frequencies = [2, 2, 1, 1, 1, 1, 0, 0]
#     size = 7
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "a", "b", "b", "c", "d", "e"]
#     assert actual == expected
#
#     values = ["a", "b", "c", "d", "e", "f", "g", "h"]
#     frequencies = [7, 1, 0, 0, 0, 0, 0, 0]
#     size = 3
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "a", "a"]
#     assert actual == expected
#
#
# # This is a little confusing.
# # It corresponds to setting record_size_ratio > 1.
# # We need to "expand" the frequencies in such a way that we still resemble
# # the source frequency distribution.
# def test_get_values_more_values_than_frequencies():
#     values = ["a", "b", "c", "d", "e", "f"]
#     frequencies = [1, 1, 1, 1]
#     size = 3
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "b", "c"]
#     assert actual == expected
#     size = 9
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "b", "c", "d", "e", "f", "a", "b", "c"]
#     assert actual == expected
#
#     values = ["a", "b", "c", "d", "e", "f", "g", "h"]
#     frequencies = [2, 2, 1]
#     size = 6
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "a", "b", "b", "c", "d"]
#     assert actual == expected
#     size = 9
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "a", "b", "b", "c", "d", "d", "e", "e"]
#     assert actual == expected
#     # Notice: we dropped f/g/h, so our actual output frequency is something like
#     # [2, 2, 2, 2, 1, 0, 0, 0]
#     # when it seems like a more closely-fitting one would be
#     # [2, 2, 2, 2, 2, 2, 1, 1]
#
#     values = ["a", "b", "c", "d", "e", "f", "g", "h"]
#     frequencies = [7, 1, 0]
#     size = 3
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "a", "a"]
#     assert actual == expected
#     size = 11
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "a", "a", "a", "a", "a", "a", "b", "d", "d", "d"]
#     assert actual == expected
#
#
# # I think this is the most confusing situation.
# # It corresponds to setting record_size_ratio < 1.
# # How do we "compress" the frequencies to apply to a smaller population?
# def test_get_values_fewer_values_than_frequencies():
#     values = ["a", "b", "c"]
#     frequencies = [1, 1, 1, 1, 1]
#     size = 3
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "b", "c"]
#     assert actual == expected
#     size = 9
#     actual = common.get_values_with_frequencies(values, frequencies, size)
#     expected = ["a", "b", "c", "a", "b", "c", "a", "b", "c"]
#     assert actual == expected
#
#     # values = ["a", "b", "c"]
#     # frequencies = [2, 2, 1, 1, 0]
#     # size = 6
#     # actual = common.get_values_with_frequencies(values, frequencies, size)
#     # expected = [???]
#     # assert actual == expected
#     # size = 9
#     # actual = common.get_values_with_frequencies(values, frequencies, size)
#     # expected = [???]
#     # assert actual == expected
#     #
#     # values = ["a", "b", "c"]
#     # frequencies = [7, 1, 1, 0, 0, 0, 0]
#     # size = 3
#     # actual = common.get_values_with_frequencies(values, frequencies, size)
#     # expected = [???]
#     # assert actual == expected
#     # size = 11
#     # actual = common.get_values_with_frequencies(values, frequencies, size)
#     # expected = [???]
#     # assert actual == expected
