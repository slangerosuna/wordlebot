use rayon::prelude::*;
use std::{
    array,
    collections::{HashMap, HashSet},
    io::Write,
};

#[derive(Debug, Clone)]
struct Constraints {
    known_letters: [Option<u8>; 5],
    included_letters: [Vec<u8>; 5],
    excluded_letters: Vec<u8>,
}

impl Constraints {
    fn new() -> Constraints {
        let known_letters = [None; 5];
        let included_letters = array::from_fn(|_| Vec::new());
        let excluded_letters = Vec::new();

        Constraints {
            known_letters,
            included_letters,
            excluded_letters,
        }
    }

    fn matches(&self, word: &str) -> bool {
        let word = word.as_bytes();

        for included_set in &self.included_letters {
            if !included_set.is_empty() && !included_set.iter().any(|&c| word.contains(&c)) {
                return false;
            }
        }

        if self.excluded_letters.iter().any(|&c| word.contains(&c)) {
            return false;
        }

        for (i, c) in word.iter().enumerate() {
            if let Some(letter) = self.known_letters[i] {
                if c != &letter {
                    return false;
                }
            } else if self.included_letters[i].contains(&c) {
                return false;
            }
        }

        true
    }

    fn update_from_guess(&mut self, guess: &str, output: [WordleAnswerColor; 5]) {
        let mut seen = HashSet::new();

        for (i, c) in guess.chars().enumerate() {
            match output[i] {
                WordleAnswerColor::Green => {
                    self.known_letters[i] = Some(c as u8);
                    seen.insert(c as u8);
                }
                WordleAnswerColor::Yellow => {
                    if !self.included_letters[i].contains(&(c as u8)) {
                        self.included_letters[i].push(c as u8);
                    }
                    seen.insert(c as u8);
                }
                WordleAnswerColor::Gray => {
                    if !self.excluded_letters.contains(&(c as u8)) && !seen.contains(&(c as u8)) {
                        self.excluded_letters.push(c as u8);
                    }
                }
            }
        }
    }
}

fn simulate_guess(correct: &str, guess: &str) -> [WordleAnswerColor; 5] {
    let mut output = [WordleAnswerColor::Gray; 5];

    for (i, c) in guess.chars().enumerate() {
        if correct.chars().nth(i) == Some(c) {
            output[i] = WordleAnswerColor::Green;
        } else if correct.contains(c) {
            output[i] = WordleAnswerColor::Yellow;
        }
    }

    output
}

fn shannon_entropy(distribution: &HashMap<[WordleAnswerColor; 5], usize>, total: usize) -> f64 {
    distribution
        .values()
        .map(|&count| {
            let p = count as f64 / total as f64;
            -p * p.log2()
        })
        .sum()
}

fn word_likelihood_score(word: &str, freq_data: &[HashMap<u8, f64>; 5]) -> f64 {
    word.as_bytes()
        .iter()
        .enumerate()
        .map(|(i, &c)| *freq_data[i].get(&c).unwrap_or(&0.))
        .sum()
}

fn find_guess_fitness(
    guess: &str,
    words: &[&str],
    probabilites: &HashMap<&str, f64>,
    freq_data: &[HashMap<u8, f64>; 5],
    seen: &SeenLetterBitFlags,
) -> f64 {
    let mut distribution = HashMap::new();

    for &word in words {
        let pattern = simulate_guess(word, guess);
        *distribution.entry(pattern).or_insert(0) += 1;
    }

    let entropy = shannon_entropy(&distribution, words.len());
    let bayesian = *probabilites.get(guess).unwrap_or(&0.0);
    let valid_bias = if words.contains(&guess) { 1.0 } else { 0.0 };
    let likelihood = word_likelihood_score(guess, freq_data) as f64;
    let seen_bias = seen.get_word(guess) as f64;

    entropy + bayesian * 2.7 + valid_bias * 0.1 + likelihood * 0.01 + seen_bias * -0.1
}

fn find_best_guess(
    all_words: &[&'static str],
    remaining_words: &[&'static str],
    probabilites: &HashMap<&str, f64>,
    freq_data: &[HashMap<u8, f64>; 5],
    seen: &SeenLetterBitFlags,
) -> &'static str {
    all_words
        .par_iter()
        .map(|&word| {
            (
                word,
                find_guess_fitness(word, remaining_words, probabilites, freq_data, seen),
            )
        })
        .max_by_key(|&guess| ordered_float::OrderedFloat(guess.1))
        .unwrap()
        .0
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum WordleAnswerColor {
    Green,
    Yellow,
    Gray,
}

fn main() {
    println!("Copyright (C) 2025 Sofia Langer-Osuna\nThis program comes with ABSOLUTELY NO WARRANTY\nThis is free software, and you are welcome to redistribute it under certain conditions.\nSee the LICENSE file for more details.\n");

    print!("Benchmark? (y/n) ");
    std::io::stdout().flush().unwrap();
    let mut bench = String::new();
    std::io::stdin().read_line(&mut bench).unwrap();
    let bench = bench.trim().to_lowercase() == "y";

    print!("Hard Mode? (y/n) ");
    std::io::stdout().flush().unwrap();
    let mut hard_mode = String::new();
    std::io::stdin().read_line(&mut hard_mode).unwrap();
    let hard_mode = hard_mode.trim().to_lowercase() == "y";

    if bench {
        benchmark(hard_mode);
    } else {
        run_assister(hard_mode);
    }
}

fn update_word_probabilities(
    words: &[&'static str],
    constraints: &Constraints,
) -> HashMap<&'static str, f64> {
    let mut probabilities = HashMap::new();

    for &word in words {
        let mut score = 1.0;
        let word_bytes = word.as_bytes();

        for (i, c) in word_bytes.iter().enumerate() {
            if let Some(known) = constraints.known_letters[i] {
                if *c == known {
                    score *= 1.5; // Boost words matching known letters
                }
            }
            if constraints.included_letters.iter().any(|v| v.contains(c)) {
                score *= 1.2; // Boost words containing useful letters
            }
            if constraints.excluded_letters.contains(c) {
                score *= 0.1; // Penalize words containing eliminated letters
            }
        }

        probabilities.insert(word, score);
    }

    // Normalize scores into probabilities
    let total: f64 = probabilities.values().sum();
    probabilities.iter_mut().for_each(|(_, v)| *v /= total);

    probabilities
}

fn letter_frequency(words: &[&str]) -> [HashMap<u8, f64>; 5] {
    let mut frequency: [HashMap<u8, f64>; 5] = Default::default();

    for &word in words {
        let bytes = word.as_bytes();
        for (i, &c) in bytes.iter().enumerate() {
            *frequency[i].entry(c).or_insert(0.) += 1.;
        }
    }

    for i in 0..5 {
        let total: f64 = frequency[i].values().sum();
        for v in frequency[i].values_mut() {
            *v /= total;
        }
    }

    frequency
}

fn benchmark(hard_mode: bool) {
    println!("Running Benchmark...");

    let mut total_iterations = 0;
    let mut failures = 0;

    let iterations = 2309;

    let all_words: Vec<&'static str> = include_str!("guess_words.txt").lines().collect();
    print!("\n\n");

    for iteration in 0..iterations {
        let mut words: Vec<&'static str> = include_str!("solution_words.txt").lines().collect();
        let mut freq_data = letter_frequency(&words);
        let correct = words[iteration];
        let mut seen = SeenLetterBitFlags::new();

        let mut constraints = Constraints::new();
        let mut probabilities: HashMap<&str, f64> = HashMap::new();

        let max_iterations = 6;
        let mut i = 0;

        loop {
            i += 1;

            let guess = if i == 1 {
                "salet" // Hard codes the first best guess because there's no point in calculating it again every time
            } else if words.len() <= 2 || i >= max_iterations {
                words.iter().max_by_key(|&word| {
                    ordered_float::OrderedFloat(*probabilities.get(*word).unwrap_or(&0.0))
                }).unwrap()
            } else if !hard_mode {
                find_best_guess(&all_words, &words, &probabilities, &freq_data, &seen)
            } else {
                find_best_guess(&words, &words, &probabilities, &freq_data, &seen)
            };
            if guess == correct {
                break;
            }

            seen.set_word(guess, true);

            let output = simulate_guess(correct, guess);
            constraints.update_from_guess(guess, output);

            words.retain(|&word| constraints.matches(word));
            probabilities = update_word_probabilities(&words, &constraints);
            freq_data = letter_frequency(&words);

            if i >= max_iterations {
                failures += 1;
                break;
            }
        }

        total_iterations += i;

        let portion_done = iteration as f32 / iterations as f32;
        let progress_bar_length = 50;
        let progress_bar = (portion_done * progress_bar_length as f32).round() as usize;
        let progress_bar_str = "=".repeat(progress_bar)
            + &if progress_bar_length > progress_bar {
                ">".to_owned() + &" ".repeat(progress_bar_length - progress_bar - 1)
            } else {
                "".to_string()
            };
        println!("\x1B[2A\r[{}]", progress_bar_str);

        println!(
            "{:.1}% done {:.1}% accuracy {:.3} average attempts",
            portion_done * 100.,
            (1. - failures as f32 / (iteration + 1) as f32) * 100.,
            total_iterations as f32 / (iteration + 1) as f32
        );
        std::io::stdout().flush().unwrap();
    }

    println!(
        "\x1B[1A\r100.0% done {:.1}% accuracy {:.3} average attempts",
        (1. - failures as f32 / iterations as f32) * 100.,
        total_iterations as f32 / iterations as f32,
    );

    if failures > 0 {
        println!("{} failures", failures);
    }
}

fn run_assister(hard_mode: bool) {
    println!("Running Assister...");
    println!("Enter your guess and the result (e.g. 'salet ggyyy') or 'exit' to quit.");
    println!("Result format: g = green, y = yellow, x = gray (e.g. 'ggyyx' for 'salet').");

    let mut probabilities: HashMap<&str, f64> = HashMap::new();
    let mut all_words: Vec<&'static str> = include_str!("guess_words.txt").lines().collect();
    let mut words: Vec<&'static str> = include_str!("solution_words.txt").lines().collect();
    let mut seen = SeenLetterBitFlags::new();
    let mut freq_data = letter_frequency(&words);
    let mut constraints = Constraints::new();
    let max_iterations = 6;
    let mut i = 0;

    loop {
        i += 1;
        let best_guess = if i == 1 {
            "salet"
        } else if words.len() <= 2 || i >= max_iterations {
            words.iter().max_by_key(|&word| {
                ordered_float::OrderedFloat(*probabilities.get(*word).unwrap_or(&0.0))
            }).unwrap()
        } else {
            find_best_guess(&all_words, &words, &probabilities, &freq_data, &seen)
        };

        println!("Best guess: {}", best_guess);

        if loop {
            std::io::stdout().flush().unwrap();
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();

            let input = input.trim();

            if input.eq_ignore_ascii_case("exit") {
                break true;
            }

            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() != 2 {
                println!("You must enter two words");
                println!("Invalid input. Please enter your guess and result (e.g. 'salet ggyyx').");
                continue;
            }
            let guess = parts[0];
            let result = parts[1];

            if guess.len() != 5 || result.len() != 5 {
                println!("Guess and result must be 5 characters long.");
                println!("Invalid input. Please enter your guess and result (e.g. 'salet ggyyx').");
                continue;
            }

            if result.chars().any(|c| !matches!(c, 'g' | 'y' | 'x')) {
                println!("Result must only contain 'g', 'y', or 'x'.");
                println!("Invalid input. Please enter your guess and result (e.g. 'salet ggyyx').");
                continue;
            }

            if !all_words.contains(&guess) {
                println!("Guess '{}' is not a valid word.", guess);
                println!("Invalid input. Please enter your guess and result (e.g. 'salet ggyyx').");
                continue;
            }

            if result == "ggggg" {
                println!("Congratulations! You've guessed the word '{}'.", guess);
                break true;
            }

            let output: Vec<WordleAnswerColor> = result
                .chars()
                .map(|c| match c {
                    'g' => WordleAnswerColor::Green,
                    'y' => WordleAnswerColor::Yellow,
                    'x' => WordleAnswerColor::Gray,
                    _ => panic!("Invalid result character: '{}'. Use 'g', 'y', or 'x'.", c),
                })
                .collect();

            constraints.update_from_guess(guess, output.try_into().unwrap());

            break false;
        } {
            break;
        }

        words.retain(|&word| constraints.matches(word));
        if hard_mode {
            all_words.retain(|&word| constraints.matches(word));
        }
        if words.is_empty() {
            println!("No valid words left. Please check your input.");
            break;
        }
        probabilities = update_word_probabilities(&words, &constraints);
        freq_data = letter_frequency(&words);
        seen.set_word(best_guess, true);
    }
}

#[derive(Copy, Clone, Debug)]
struct SeenLetterBitFlags(u32);

impl SeenLetterBitFlags {
    fn new() -> Self {
        Self(0)
    }

    fn flag_of_char(c: char) -> u32 {
        0x1 << ((c as u8) - 'a' as u8)
    }

    fn set(&mut self, c: char, val: bool) {
        let flag = Self::flag_of_char(c);
        if val {
            self.0 |= flag;
        } else {
            self.0 &= !flag;
        }
    }

    fn set_word(&mut self, w: &str, val: bool) {
        for c in w.chars() {
            self.set(c, val);
        }
    }

    fn get(&self, c: char) -> bool {
        let flag = Self::flag_of_char(c);
        self.0 & flag != 0
    }

    fn get_word(&self, w: &str) -> usize {
        w.chars().filter(|c| self.get(*c)).count()
    }
}
