#ifndef TEST_CORRECTNESS_H
#define TEST_CORRECTNESS_H

#include "Matrix.h"
#include <iostream>
#include <string>

/**
 * @brief Print result of a test (PASS or FAIL).
 * @param testName Name of the test.
 *CH @param success True if the test is correct, else false.
 */
void printTestResult(const std::string& testName, bool success);

/**
 * @brief Test 1: Compare the results of Naive, Strassen, và Cannon.
 * Naive is ground truth.
 * @param size Matrix size (sizexsize).
 * @return True if all algorithm return the same result, else false.
 */
bool testAlgorithmComparison(int size);

/**
 * @brief Test 2: Test with Identity Matrix.
 * A * I = A
 * @param size Matrix size (sizexsize).
 * @return True if A * I == A for both Naive and Strassen.
 */
bool testIdentityMultiplication(int size);

/**
 * @brief Test 3: Test with Zero Matrix.
 * A * 0 = 0
 * @param size Matrix size (sizexsize).
 * @return True if A * 0 == 0 for both Naive and Strassen.
 */
bool testZeroMultiplication(int size);

/**
 * @brief Main function run all the tests.
 */
void runFullTestSuite();

#endif // TEST_CORRECTNESS_H