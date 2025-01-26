use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Serialize, Deserialize)]
struct TokenizerOutput {
    tokens: Vec<String>,
    ids: Vec<u32>,
}

struct TurkishTokenizer {
    roots: HashMap<String, u32>,
    suffixes: HashMap<String, u32>,
    bpe_tokens: HashMap<String, u32>,
    word_regex: Regex,
}

impl TurkishTokenizer {
    fn new() -> Self {
        let roots = Self::load_json("kokler_v04.json");
        let suffixes = Self::load_json("ekler_v04.json");
        let bpe_tokens = Self::load_json("bpe_v02.json");
        let word_regex = Regex::new(r"[\w]+|[.,!?;]").unwrap();

        TurkishTokenizer {
            roots,
            suffixes,
            bpe_tokens,
            word_regex,
        }
    }

    fn load_json(file_path: &str) -> HashMap<String, u32> {
        let file = File::open(file_path).expect(&format!("Failed to open {}", file_path));
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).expect(&format!("Failed to parse {}", file_path))
    }

    fn tokenize(&self, text: &str) -> TokenizerOutput {
        let words: Vec<_> = self.word_regex.find_iter(text).collect();
        let tokens_and_ids: Vec<(Vec<String>, Vec<u32>)> = words
            .par_iter() // Parallelize processing while maintaining order
            .map(|word| self.process_word(word.as_str()))
            .collect();

        let mut tokens = Vec::new();
        let mut ids = Vec::new();

        for (token_list, id_list) in tokens_and_ids {
            tokens.extend(token_list);
            ids.extend(id_list);
        }

        TokenizerOutput { tokens, ids }
    }

    fn process_word(&self, word: &str) -> (Vec<String>, Vec<u32>) {
        let mut tokens = Vec::new();
        let mut ids = Vec::new();

        if word.chars().any(char::is_uppercase) {
            // Add initial UPCL token if word starts with uppercase
            if word.chars().next().unwrap().is_uppercase() {
                tokens.push("<UPCL>".to_string());
                ids.push(*self.roots.get("<UPCL>").unwrap_or(&0));
            }

            // Split by uppercase letters and process each part
            let mut current = String::new();
            let mut is_first = true;

            for c in word.chars() {
                if c.is_uppercase() && !is_first {
                    if !current.is_empty() {
                        self.process_lowercase_word(&current.to_lowercase(), &mut tokens, &mut ids);
                    }
                    tokens.push("<UPCL>".to_string());
                    ids.push(*self.roots.get("<UPCL>").unwrap_or(&0));
                    current.clear();
                    current.push(c);
                } else {
                    current.push(c);
                }
                is_first = false;
            }

            if !current.is_empty() {
                self.process_lowercase_word(&current.to_lowercase(), &mut tokens, &mut ids);
            }
        } else {
            self.process_lowercase_word(word, &mut tokens, &mut ids);
        }

        (tokens, ids)
    }

    fn process_lowercase_word(&self, word: &str, tokens: &mut Vec<String>, ids: &mut Vec<u32>) {
        if let Some((root, root_id, remainder)) = self.match_root(word) {
            tokens.push(root);
            ids.push(root_id);
            self.process_remainder(&remainder, tokens, ids);
        } else {
            self.process_bpe(word, tokens, ids);
        }
    }

    fn match_root(&self, word: &str) -> Option<(String, u32, String)> {
        let chars: Vec<char> = word.chars().collect();
        for i in (1..=chars.len()).rev() {
            let prefix: String = chars[..i].iter().collect();
            if let Some(&id) = self.roots.get(&prefix) {
                let remainder: String = chars[i..].iter().collect();
                return Some((prefix, id, remainder));
            }
        }
        None
    }

    fn process_remainder(&self, remainder: &str, tokens: &mut Vec<String>, ids: &mut Vec<u32>) {
        if remainder.is_empty() {
            return;
        }

        if let Some((suffix, suffix_id)) = self.match_suffix(remainder) {
            tokens.push(suffix.clone());
            ids.push(suffix_id);
            let new_remainder = &remainder[suffix.len()..];
            self.process_remainder(new_remainder, tokens, ids);
        } else {
            self.process_bpe(remainder, tokens, ids);
        }
    }

    fn match_suffix(&self, word: &str) -> Option<(String, u32)> {
        let chars: Vec<char> = word.chars().collect();
        for i in (1..=chars.len()).rev() {
            let current: String = chars[..i].iter().collect();
            if let Some(&id) = self.suffixes.get(&current) {
                return Some((current, id));
            }
        }
        None
    }

    fn process_bpe(&self, word: &str, tokens: &mut Vec<String>, ids: &mut Vec<u32>) {
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut found = false;

            for j in (i + 1..=chars.len()).rev() {
                let substr: String = chars[i..j].iter().collect();
                if let Some(&id) = self.bpe_tokens.get(&substr) {
                    tokens.push(substr);
                    ids.push(id);
                    i = j;
                    found = true;
                    break;
                }
            }

            if !found {
                i += 1;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_text>", args[0]);
        std::process::exit(1);
    }

    let tokenizer = TurkishTokenizer::new();
    let input = args[1..].join(" ");
    let output = tokenizer.tokenize(&input);

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
