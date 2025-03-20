use futures::future::join_all;
use rand::RngCore;
use std::array;
use tokio::runtime::Runtime;

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
            if included_set.len() != 0 && !included_set.iter().any(|&c| word.contains(&c)) {
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
        for (i, c) in guess.chars().enumerate() {
            match output[i] {
                WordleAnswerColor::Green => {
                    self.known_letters[i] = Some(c as u8);
                }
                WordleAnswerColor::Yellow => {
                    if !self.included_letters[i].contains(&(c as u8)) {
                        self.included_letters[i].push(c as u8);
                    }
                }
                WordleAnswerColor::Gray => {
                    if !self.excluded_letters.contains(&(c as u8)) {
                        self.excluded_letters.push(c as u8);
                    }
                }
            }
        }
    }

    fn from_guess_answer_color(guess: &str, output: [WordleAnswerColor; 5]) -> Constraints {
        let mut constraints = Constraints::new();
        constraints.update_from_guess(guess, output);
        constraints
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

fn amount_eliminated(correct: &str, guess: &str, words: &[&str]) -> usize {
    let len = words.len();
    const MAX_SAMPLE: usize = 500;

    let guess_output = simulate_guess(correct, guess);
    let constraints = Constraints::from_guess_answer_color(guess, guess_output);

    if len <= MAX_SAMPLE {
        let mut count = 0;

        for word in words {
            if !constraints.matches(word) {
                count += 1;
            }
        }

        count
    } else {
        let sampled = rand::seq::index::sample(&mut rand::rng(), len, MAX_SAMPLE);

        let mut count = 0;

        for sample in sampled {
            let word = words[sample];
            if !constraints.matches(word) {
                count += 1;
            }
        }

        count
    }
}

fn find_guess_fitness(guess: &str, words: &[&str]) -> usize {
    words
        .iter()
        .map(|&word| amount_eliminated(word, guess, words))
        .sum()
}

async fn find_best_guess(all_words: &[&'static str], remaining_words: &[&'static str]) -> &'static str {
    // make words live forever
    let remaining_words = unsafe { std::mem::transmute::<&[&'static str], &'static [&'static str]>(remaining_words) };
    let all_words = unsafe { std::mem::transmute::<&[&'static str], &'static [&'static str]>(all_words) };

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
        .max_by_key(|&guess| guess.1)
        .unwrap()
        .0
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
enum WordleAnswerColor {
    Green,
    Yellow,
    Gray,
}

fn main() {
    let rt = Runtime::new().unwrap();

    let mut total_iterations = 0;
    let mut failures = 0;

    let iterations = 1;

    let all_words: Vec<&'static str> = include_str!("guess_words.txt").lines().collect();

    for iteration in 0..iterations {
        let mut words: Vec<&'static str> = include_str!("solution_words.txt").lines().collect();
        let correct = rand::rng().next_u32() as usize % words.len();
        let correct = "spark"; // words[correct];

        let mut constraints = Constraints::new();

        let max_iterations = 6;
        let mut i = 0;

        loop {
            i += 1;

            let now = std::time::Instant::now();
            let guess = if i == 1 {
                "raise"
            } else if words.len() <= 2 {
                words[0]
            } else {
                rt.block_on(find_best_guess(&all_words, &words))
            };
            if guess == correct {
                break;
            }

            let output= simulate_guess(correct, guess);
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

        //display every 5% of the way through
        if iterations > 20 && iteration % (iterations / 20) == 0 {
            println!("{}% done", iteration * 100 / iterations);
        }
    }

    println!("Average iterations: {}", total_iterations as f32 / iterations as f32);
    println!("Success rate {}%", (1. - failures as f32 / iterations as f32) * 100.);
}
