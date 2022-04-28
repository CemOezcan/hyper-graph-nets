import unittest
from argparse import ArgumentParser, Namespace
from util.Types import *


def main():
    console_arguments = get_arguments()
    run_tests(console_arguments=console_arguments)


def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-w', '--whitelist', nargs='+', default=[])
    parser.add_argument('-b', '--blacklist', nargs='+', default=[])
    args = parser.parse_args()
    assert len(args.whitelist) == 0 or len(args.blacklist) == 0, "Can not provide blacklist and whitelist"
    return args


def run_tests(console_arguments: Namespace):
    all_tests = unittest.TestLoader().discover(start_dir="tests", pattern="*")
    print(all_tests)

    if console_arguments.blacklist:  # provided a blacklist
        filtered_tests = blacklist_tests(test_suite=all_tests, blacklist=console_arguments.blacklist)
    elif console_arguments.whitelist:  # provided a whitelist
        filtered_tests = whitelist_tests(test_suite=all_tests, whitelist=console_arguments.whitelist)
    else:
        filtered_tests = all_tests

    unittest.TextTestRunner().run(filtered_tests)  # run all tests and print the results as a text


def blacklist_tests(test_suite: unittest.TestSuite, blacklist: List[str]) -> unittest.TestSuite:
    selected_tests = unittest.TestSuite()
    for test_group in test_suite._tests:
        if isinstance(test_group, unittest.TestSuite):  # recurse
            selected_tests.addTest(blacklist_tests(test_suite=test_group, blacklist=blacklist))
        elif not _test_case_in_list(test_case=test_group, list_of_names=blacklist):
            selected_tests.addTest(test_group)
    return selected_tests


def whitelist_tests(test_suite: unittest.TestSuite, whitelist: List[str]) -> unittest.TestSuite:
    selected_tests = unittest.TestSuite()
    for test_group in test_suite._tests:
        if isinstance(test_group, unittest.TestSuite):  # recurse
            selected_tests.addTest(whitelist_tests(test_suite=test_group, whitelist=whitelist))
        elif _test_case_in_list(test_case=test_group, list_of_names=whitelist):
            selected_tests.addTest(test_group)
    return selected_tests


def _test_case_in_list(test_case: unittest.TestCase, list_of_names: List[str]) -> bool:
    class_hierarchy = str(test_case.__class__).split("'")[1].split(".")
    class_hierarchy.append(test_case._testMethodName)
    return any(x in list_of_names for x in class_hierarchy)


if __name__ == '__main__':
    main()
