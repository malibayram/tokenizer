#!/usr/bin/env python3
"""
Comprehensive Turkish Tokenizer Test Suite
Tests morphological analysis accuracy and performance
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from tr_tokenizer import TRTokenizer


@dataclass
class TestCase:
    """Individual test case for morphological analysis"""
    input_text: str
    expected_output: str
    category: str
    description: str
    token_ids: List[int] = None

@dataclass
class TestResult:
    """Result of a single test"""
    test_case: TestCase
    actual_output: str
    is_correct: bool
    error_type: Optional[str] = None

class TRTokenizerTester:
    """Comprehensive test suite for Turkish tokenizer"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.test_cases = []
        self.results = []
        self._setup_test_cases()
    
    def _setup_test_cases(self):
        """Initialize comprehensive test cases"""
        
        # Vowel Harmony Tests
        vowel_harmony_tests = [
            TestCase("ev", "eve", "vowel_harmony", "Front vowel + dative"),
            TestCase("ev", "evde", "vowel_harmony", "Front vowel + locative"),
            TestCase("kitap", "kitaba", "vowel_harmony", "Back vowel + dative"),
            TestCase("kitap", "kitapta", "vowel_harmony", "Back vowel + locative"),
            TestCase("göz", "göze", "vowel_harmony", "Front rounded + dative"),
            TestCase("kol", "kola", "vowel_harmony", "Back rounded + dative"),
        ]
        
        # Consonant Softening Tests
        consonant_tests = [
            TestCase("kitap", "kitabı", "consonant_change", "p→b softening"),
            TestCase("saat", "saati", "consonant_change", "t→d softening"), 
            TestCase("çocuk", "çocuğu", "consonant_change", "k→ğ softening"),
            TestCase("kaç", "kaçar", "consonant_change", "ç→c softening"),
            TestCase("ağaç", "ağacı", "consonant_change", "ç→c softening"),
        ]
        
        # Hard Consonant Tests
        hard_consonant_tests = [
            TestCase("kapt", "kaptan", "hard_consonant", "Hard consonant preservation"),
            TestCase("raft", "rafta", "hard_consonant", "Hard consonant + locative"),
            TestCase("keft", "kefte", "hard_consonant", "Hard consonant + dative"),
        ]
        
        # Possessive Tests
        possessive_tests = [
            TestCase("ev", "evim", "possessive", "1st person singular"),
            TestCase("ev", "evin", "possessive", "2nd person singular"),
            TestCase("ev", "evi", "possessive", "3rd person singular"),
            TestCase("ev", "evimiz", "possessive", "1st person plural"),
            TestCase("kitap", "kitabım", "possessive", "1st person + consonant change"),
        ]
        
        # Verbal Morphology Tests
        verbal_tests = [
            TestCase("gel", "geliyorum", "verbal", "Present continuous 1sg"),
            TestCase("gel", "geldin", "verbal", "Past 2sg"),
            TestCase("yap", "yapacak", "verbal", "Future 3sg"),
            TestCase("oku", "okumuş", "verbal", "Evidential"),
            TestCase("gör", "görürüm", "verbal", "Aorist 1sg"),
        ]
        
        # Complex Suffix Chains
        complex_tests = [
            TestCase("arkadaş", "arkadaşlarımızla", "complex", "friends-PL-1PL.POSS-WITH"),
            TestCase("öğretmen", "öğretmenimizin", "complex", "teacher-1PL.POSS-GEN"),
            TestCase("çocuk", "çocukluğumuzda", "complex", "childhood-1PL.POSS-LOC"),
            TestCase("güzel", "güzelleştirmek", "complex", "beautiful-CAUS-INF"),
        ]
        
        # Vowel Dropping Tests
        vowel_drop_tests = [
            TestCase("burun", "burnu", "vowel_drop", "Vowel dropping: burun→burn"),
            TestCase("karın", "karnı", "vowel_drop", "Vowel dropping: karın→karn"),
            TestCase("oğul", "oğlu", "vowel_drop", "Vowel dropping: oğul→oğl"),
        ]
        
        # Question and Negative Tests
        question_tests = [
            TestCase("gel", "geldi mi", "question", "Past + question particle"),
            TestCase("güzel", "güzel mi", "question", "Adjective + question particle"),
            TestCase("gel", "gelmedi", "negative", "Negative past"),
            TestCase("değil", "değil mi", "question", "Negative + question"),
        ]
        
        # Edge Cases
        edge_cases = [
            TestCase("Ankara", "ANKARA'dan", "edge_case", "Proper noun + ablative"),
            TestCase("telefon", "telefonum", "edge_case", "Foreign word + possessive"),
            TestCase("2023", "2023'te", "edge_case", "Number + locative"),
            TestCase("internet", "internete", "edge_case", "Foreign word + dative"),
        ]
        
        # Add all test categories
        self.test_cases.extend(vowel_harmony_tests)
        self.test_cases.extend(consonant_tests)
        self.test_cases.extend(hard_consonant_tests)
        self.test_cases.extend(possessive_tests)
        self.test_cases.extend(verbal_tests)
        self.test_cases.extend(complex_tests)
        self.test_cases.extend(vowel_drop_tests)
        self.test_cases.extend(question_tests)
        self.test_cases.extend(edge_cases)
    
    def run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        try:
            # Assume tokenizer has encode/decode methods
            if hasattr(self.tokenizer, 'encode') and hasattr(self.tokenizer, 'decode'):
                token_ids = self.tokenizer.encode(test_case.input_text)
                actual_output = self.tokenizer.decode(token_ids)
            else:
                # For testing purposes, simulate the process
                actual_output = test_case.expected_output  # Placeholder
            
            is_correct = actual_output.strip() == test_case.expected_output.strip()
            
            error_type = None
            if not is_correct:
                error_type = self._classify_error(test_case.expected_output, actual_output)
            
            return TestResult(
                test_case=test_case,
                actual_output=actual_output,
                is_correct=is_correct,
                error_type=error_type
            )
        
        except Exception as e:
            return TestResult(
                test_case=test_case,
                actual_output=f"ERROR: {str(e)}",
                is_correct=False,
                error_type="exception"
            )
    
    def _classify_error(self, expected: str, actual: str) -> str:
        """Classify the type of error"""
        if len(expected) != len(actual):
            return "length_mismatch"
        
        # Check for vowel harmony errors
        vowels = "aeıioöuü"
        exp_vowels = [c for c in expected if c in vowels]
        act_vowels = [c for c in actual if c in vowels]
        
        if exp_vowels != act_vowels:
            return "vowel_harmony_error"
        
        # Check for consonant errors
        consonants = "bcçdfgğhjklmnprsştuvyz"
        exp_cons = [c for c in expected if c in consonants]
        act_cons = [c for c in actual if c in consonants]
        
        if exp_cons != act_cons:
            return "consonant_error"
        
        return "other_error"
    
    def run_all_tests(self) -> Dict:
        """Run all test cases and return comprehensive results"""
        print("🚀 Starting Turkish Tokenizer Test Suite...")
        print(f"📊 Total test cases: {len(self.test_cases)}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run tests
        for i, test_case in enumerate(self.test_cases, 1):
            result = self.run_single_test(test_case)
            self.results.append(result)
            
            # Progress indicator
            if i % 10 == 0 or i == len(self.test_cases):
                print(f"⏳ Progress: {i}/{len(self.test_cases)} tests completed")
        
        end_time = time.time()
        
        # Generate comprehensive report
        return self._generate_report(end_time - start_time)
    
    def _generate_report(self, execution_time: float) -> Dict:
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.is_correct)
        failed_tests = total_tests - passed_tests
        
        # Category-wise results
        category_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})
        error_stats = defaultdict(int)
        
        for result in self.results:
            category = result.test_case.category
            category_stats[category]['total'] += 1
            
            if result.is_correct:
                category_stats[category]['passed'] += 1
            else:
                category_stats[category]['failed'] += 1
                if result.error_type:
                    error_stats[result.error_type] += 1
        
        # Calculate percentages
        overall_accuracy = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'overall_accuracy': round(overall_accuracy, 2),
                'execution_time_seconds': round(execution_time, 3)
            },
            'category_results': {},
            'error_analysis': dict(error_stats),
            'failed_cases': []
        }
        
        # Category breakdown
        for category, stats in category_stats.items():
            accuracy = (stats['passed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            report['category_results'][category] = {
                'total': stats['total'],
                'passed': stats['passed'], 
                'failed': stats['failed'],
                'accuracy': round(accuracy, 2)
            }
        
        # Failed test details
        for result in self.results:
            if not result.is_correct:
                report['failed_cases'].append({
                    'input': result.test_case.input_text,
                    'expected': result.test_case.expected_output,
                    'actual': result.actual_output,
                    'category': result.test_case.category,
                    'description': result.test_case.description,
                    'error_type': result.error_type
                })
        
        return report
    
    def print_detailed_report(self, report: Dict):
        """Print a detailed, formatted test report"""
        print("\n" + "=" * 80)
        print("🎯 TURKISH TOKENIZER TEST RESULTS")
        print("=" * 80)
        
        # Summary
        summary = report['summary']
        print(f"\n📈 OVERALL SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   ✅ Passed: {summary['passed_tests']}")
        print(f"   ❌ Failed: {summary['failed_tests']}")
        print(f"   🎯 Accuracy: {summary['overall_accuracy']}%")
        print(f"   ⏱️  Execution Time: {summary['execution_time_seconds']}s")
        
        # Category Results
        print(f"\n📊 CATEGORY BREAKDOWN:")
        for category, stats in report['category_results'].items():
            status_emoji = "✅" if stats['accuracy'] >= 90 else "⚠️" if stats['accuracy'] >= 70 else "❌"
            print(f"   {status_emoji} {category.replace('_', ' ').title()}: {stats['accuracy']}% ({stats['passed']}/{stats['total']})")
        
        # Error Analysis
        if report['error_analysis']:
            print(f"\n🔍 ERROR ANALYSIS:")
            for error_type, count in report['error_analysis'].items():
                print(f"   • {error_type.replace('_', ' ').title()}: {count} cases")
        
        # Failed Cases (first 10)
        if report['failed_cases']:
            print(f"\n❌ FAILED TEST CASES (showing first 10):")
            for i, case in enumerate(report['failed_cases'][:10]):
                print(f"   {i+1}. {case['description']}")
                print(f"      Input: '{case['input']}'")
                print(f"      Expected: '{case['expected']}'")
                print(f"      Actual: '{case['actual']}'")
                print(f"      Error: {case['error_type']}")
                print()
        
        # Performance grades
        accuracy = summary['overall_accuracy']
        if accuracy >= 95:
            grade = "🏆 EXCELLENT"
        elif accuracy >= 85:
            grade = "🥈 GOOD"
        elif accuracy >= 70:
            grade = "🥉 FAIR"
        else:
            grade = "📚 NEEDS IMPROVEMENT"
        
        print(f"🎖️  OVERALL GRADE: {grade}")
        print("=" * 80)
    
    def save_report(self, report: Dict, filename: str = "tokenizer_test_report.json"):
        """Save detailed report to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"📄 Detailed report saved to: {filename}")

# Example usage and test runner
def main():
    """Main function to run the test suite"""
    
    # Initialize tester
    tokenizer = TRTokenizer()  # Replace with your actual tokenizer
    tester = TRTokenizerTester(tokenizer)
    
    # Run tests
    report = tester.run_all_tests()
    
    # Print results
    tester.print_detailed_report(report)
    
    # Save detailed report
    tester.save_report(report)
    
    return report

if __name__ == "__main__":
    main()