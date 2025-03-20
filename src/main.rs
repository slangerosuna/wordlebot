use futures::future::join_all;
use std::{array, collections::{HashMap, HashSet}, io::Write};
use tokio::runtime::*;

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

fn find_guess_fitness(guess: &str, words: &[&str]) -> f64 {
    let mut distribution = HashMap::new();

    for &word in words {
        let pattern = simulate_guess(word, guess);
        *distribution.entry(pattern).or_insert(0) += 1;
    }

    let entropy = shannon_entropy(&distribution, words.len());

    entropy + if words.contains(&guess) { 0.01 } else { 0.0 } // Small bias for valid words
}

async fn find_best_guess(
    all_words: &[&'static str],
    remaining_words: &[&'static str],
) -> &'static str {
    // make words live forever
    let remaining_words =
        unsafe { std::mem::transmute::<&[&'static str], &'static [&'static str]>(remaining_words) };
    let all_words =
        unsafe { std::mem::transmute::<&[&'static str], &'static [&'static str]>(all_words) };

    let batch_size = (all_words.len() / num_cpus::get()).max(1) * 2;
    let futures = all_words.chunks(batch_size).map(|chunk| {
        tokio::spawn(async move {
            chunk
                .iter()
                .map(|&word| (word, find_guess_fitness(word, remaining_words)))
                .collect::<Vec<_>>()
        })
    });

    join_all(futures)
        .await
        .iter()
        .flatten()
        .flatten()
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
    let rt = Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()
        .unwrap();

    let mut total_iterations = 0;
    let mut failures = 0;

    let iterations = 2309;

    let all_words: Vec<&'static str> = include_str!("guess_words.txt").lines().collect();
    print!("\n\n");

    for iteration in 1..iterations {
        let mut words: Vec<&'static str> = include_str!("solution_words.txt").lines().collect();
        //let correct = rand::rng().next_u32() as usize % words.len();
        let correct = words[iteration];

        let mut constraints = Constraints::new();

        let max_iterations = 6;
        let mut i = 0;

        loop {
            i += 1;

            let guess = if i == 1 {
                "salet"
            } else if words.len() <= 2 || i >= max_iterations {
                words[0]
            } else {
                rt.block_on(find_best_guess(&all_words, &words))
            };
            if guess == correct {
                break;
            }

            let output = simulate_guess(correct, guess);
            constraints.update_from_guess(guess, output);

            words = words
                .into_iter()
                .filter(|&word| constraints.matches(word))
                .collect();

            if i >= max_iterations {
                failures += 1;
                break;
            }
        }

        total_iterations += i;

        let portion_done = iteration as f32 / iterations as f32;
        // 25 characters for the progress bar
        let progress_bar_length = 50;
        let progress_bar = (portion_done * progress_bar_length as f32).round() as usize;
        let progress_bar_str = "=".repeat(progress_bar) + if progress_bar_length > progress_bar { ">" } else { "" } + &" ".repeat(progress_bar_length - progress_bar - 1);
        println!("\x1B[2A\r[{}]", progress_bar_str);

        println!(
            "{:.1}% done {:.3}% accuracy {:.3} average attempts",
            portion_done * 100.,
            (1. - failures as f32 / (iteration + 1) as f32) * 100.,
            total_iterations as f32 / (iteration + 1) as f32
        );
        std::io::stdout().flush().unwrap();
    }

    println!(
        "\x1B[1A\r100.0% done {:.0}% accuracy {:.3} average attempts",
        (1. - failures as f32 / iterations as f32) * 100.,
        total_iterations as f32 / iterations as f32,
    );

    if failures > 0 {
        println!("{} failures", failures);
    }
}
