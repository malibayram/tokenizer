use lazy_static::lazy_static;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

// Static references for special tokens
static UPPERCASE_TOKEN: &str = "<uppercase>";
static SPACE_TOKEN: &str = "<space>";
static NEWLINE_TOKEN: &str = "<newline>";
static TAB_TOKEN: &str = "<tab>";
static UNKNOWN_TOKEN: &str = "<unknown>";

// Special token IDs
const UPPERCASE_ID: u32 = 0;
const SPACE_ID: u32 = 1;
const NEWLINE_ID: u32 = 2;
const TAB_ID: u32 = 3;
const UNKNOWN_ID: u32 = 4;

lazy_static! {
    static ref WORD_BOUNDARY: Regex = Regex::new(r"[\p{L}\p{N}]+|[.,!?;]").unwrap();
    static ref UPPERCASE_SPLIT: Regex = Regex::new(r"([A-Z][^A-Z\s]*)|([^A-Z\s]+)").unwrap();
}

#[derive(Debug, Serialize, Deserialize)]
struct TokenizerOutput {
    tokens: Vec<String>,
    ids: Vec<u32>,
}

#[derive(Debug)]
pub enum TokenizerError {
    FileNotFound(String),
    ParseError(String),
    InvalidInput(String),
}

struct TokenCache {
    roots: Arc<HashMap<String, u32>>,
    suffixes: Arc<HashMap<String, u32>>,
    bpe_tokens: Arc<HashMap<String, u32>>,
    lookup_cache: HashMap<String, Option<(String, u32, String)>>,
}

struct TurkishTokenizer {
    cache: TokenCache,
}

impl TurkishTokenizer {
    fn new() -> Result<Self, TokenizerError> {
        let current_dir = std::env::current_dir().map_err(|e| TokenizerError::FileNotFound(e.to_string()))?;
        let roots = Arc::new(Self::load_json(current_dir.join("kokler_v05.json"))?);
        let suffixes = Arc::new(Self::load_json(current_dir.join("ekler_v05.json"))?);
        let bpe_tokens = Arc::new(Self::load_json(current_dir.join("bpe_v05.json"))?);

        Ok(TurkishTokenizer {
            cache: TokenCache {
                roots,
                suffixes,
                bpe_tokens,
                lookup_cache: HashMap::new(),
            },
        })
    }

    fn load_json<P: AsRef<Path>>(file_path: P) -> Result<HashMap<String, u32>, TokenizerError> {
        let file = File::open(&file_path)
            .map_err(|e| TokenizerError::FileNotFound(format!("Failed to open {:?}: {}", file_path.as_ref(), e)))?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| TokenizerError::ParseError(format!("Failed to parse {:?}: {}", file_path.as_ref(), e)))
    }

    fn tokenize(&mut self, text: &str) -> Result<TokenizerOutput, TokenizerError> {
        let text = text.replace("\\n", "\n").replace("\\t", "\t");
        let mut tokens = Vec::new();
        let mut ids = Vec::new();

        // Split text into chunks for parallel processing
        let chunks: Vec<_> = WORD_BOUNDARY.find_iter(&text)
            .collect();

        // Process chunks in parallel
        let results: Vec<_> = chunks.par_iter()
            .map(|m| {
                let mut chunk_tokens = Vec::new();
                let mut chunk_ids = Vec::new();
                let word = m.as_str();

                if word.chars().any(char::is_uppercase) {
                    // Process uppercase words
                    for cap in UPPERCASE_SPLIT.captures_iter(word) {
                        if let Some(part) = cap.get(0) {
                            let part = part.as_str();
                            if !part.is_empty() {
                                if part.chars().next().unwrap().is_uppercase() {
                                    chunk_tokens.push(UPPERCASE_TOKEN.to_string());
                                    chunk_ids.push(UPPERCASE_ID);
                                    self.process_lowercase_word(&part.to_lowercase(), &mut chunk_tokens, &mut chunk_ids)?;
                                } else {
                                    self.process_lowercase_word(part, &mut chunk_tokens, &mut chunk_ids)?;
                                }
                            }
                        }
                    }
                } else {
                    self.process_lowercase_word(word, &mut chunk_tokens, &mut chunk_ids)?;
                }

                Ok((chunk_tokens, chunk_ids))
            })
            .collect::<Result<Vec<_>, TokenizerError>>()?;

        // Combine results and handle whitespace
        let mut last_end = 0;
        for (i, m) in chunks.iter().enumerate() {
            // Handle whitespace before the chunk
            for c in text[last_end..m.start()].chars() {
                match c {
                    ' ' => {
                        tokens.push(SPACE_TOKEN.to_string());
                        ids.push(SPACE_ID);
                    }
                    '\n' => {
                        tokens.push(NEWLINE_TOKEN.to_string());
                        ids.push(NEWLINE_ID);
                    }
                    '\t' => {
                        tokens.push(TAB_TOKEN.to_string());
                        ids.push(TAB_ID);
                    }
                    _ => {
                        tokens.push(UNKNOWN_TOKEN.to_string());
                        ids.push(UNKNOWN_ID);
                    }
                }
            }

            // Add chunk results
            let (chunk_tokens, chunk_ids) = &results[i];
            tokens.extend(chunk_tokens.iter().cloned());
            ids.extend_from_slice(chunk_ids);

            last_end = m.end();
        }

        // Handle remaining whitespace
        for c in text[last_end..].chars() {
            match c {
                ' ' => {
                    tokens.push(SPACE_TOKEN.to_string());
                    ids.push(SPACE_ID);
                }
                '\n' => {
                    tokens.push(NEWLINE_TOKEN.to_string());
                    ids.push(NEWLINE_ID);
                }
                '\t' => {
                    tokens.push(TAB_TOKEN.to_string());
                    ids.push(TAB_ID);
                }
                _ => {
                    tokens.push(UNKNOWN_TOKEN.to_string());
                    ids.push(UNKNOWN_ID);
                }
            }
        }

        Ok(TokenizerOutput { tokens, ids })
    }

    fn process_lowercase_word(&self, word: &str, tokens: &mut Vec<String>, ids: &mut Vec<u32>) -> Result<(), TokenizerError> {
        if let Some((root, root_id, remainder)) = self.match_root_cached(word) {
            tokens.push(root.to_string());
            ids.push(root_id);
            if !remainder.is_empty() {
                self.process_remainder(remainder, tokens, ids)?;
            }
        } else {
            let success = self.process_bpe(word, tokens, ids);
            if !success {
                tokens.push(UNKNOWN_TOKEN.to_string());
                ids.push(UNKNOWN_ID);
            }
        }
        Ok(())
    }

    fn match_root_cached<'a>(&'a self, word: &'a str) -> Option<(&'a str, u32, &'a str)> {
        if let Some(cached) = self.cache.lookup_cache.get(word) {
            return cached.as_ref().map(|(r, id, rem)| (r.as_str(), *id, rem.as_str()));
        }
        self.match_root(word)
    }

    fn match_root<'a>(&'a self, word: &'a str) -> Option<(&'a str, u32, &'a str)> {
        for (i, _) in word.char_indices().rev() {
            let (prefix, remainder) = word.split_at(i);
            if let Some(&id) = self.cache.roots.get(prefix) {
                return Some((prefix, id, remainder));
            }
        }
        None
    }

    fn process_remainder(&self, remainder: &str, tokens: &mut Vec<String>, ids: &mut Vec<u32>) -> Result<(), TokenizerError> {
        if let Some((suffix, suffix_id)) = self.match_suffix(remainder) {
            tokens.push(suffix.to_string());
            ids.push(suffix_id);
            let new_remainder = &remainder[suffix.len()..];
            if !new_remainder.is_empty() {
                self.process_remainder(new_remainder, tokens, ids)?;
            }
        } else if let Some((root, root_id, new_remainder)) = self.match_root_cached(remainder) {
            tokens.push(root.to_string());
            ids.push(root_id);
            if !new_remainder.is_empty() {
                self.process_remainder(new_remainder, tokens, ids)?;
            }
        } else {
            let success = self.process_bpe(remainder, tokens, ids);
            if !success {
                tokens.push(UNKNOWN_TOKEN.to_string());
                ids.push(UNKNOWN_ID);
            }
        }
        Ok(())
    }

    fn match_suffix<'a>(&'a self, word: &'a str) -> Option<(&'a str, u32)> {
        for (i, _) in word.char_indices().rev() {
            let current = &word[..i];
            if let Some(&id) = self.cache.suffixes.get(current) {
                return Some((current, id));
            }
        }
        None
    }

    fn process_bpe(&self, word: &str, tokens: &mut Vec<String>, ids: &mut Vec<u32>) -> bool {
        let mut i = 0;
        let mut found_any = false;

        while i < word.len() {
            let mut found = false;
            let mut max_len = word.len();

            // Try to find the longest matching substring first
            while max_len > i {
                if let Some(substr) = word.get(i..max_len) {
                    if let Some(&id) = self.cache.bpe_tokens.get(substr) {
                        tokens.push(substr.to_string());
                        ids.push(id);
                        i = max_len;
                        found = true;
                        found_any = true;
                        break;
                    }
                }
                max_len -= 1;
            }

            if !found {
                i += 1;
            }
        }
        found_any
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_text>", args[0]);
        std::process::exit(1);
    }

    let mut tokenizer = match TurkishTokenizer::new() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to initialize tokenizer: {:?}", e);
            std::process::exit(1);
        }
    };

    let input = args[1..].join(" ");
    match tokenizer.tokenize(&input) {
        Ok(output) => println!("{}", serde_json::to_string_pretty(&output).unwrap()),
        Err(e) => {
            eprintln!("Failed to tokenize input: {:?}", e);
            std::process::exit(1);
        }
    }
}
