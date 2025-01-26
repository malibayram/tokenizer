use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct TokenizerOutput {
    tokens: Vec<String>,
    ids: Vec<u32>,
}

struct TurkishTokenizer {
    roots: HashMap<String, u32>,
    suffixes: HashMap<String, u32>,
    bpe_tokens: HashMap<String, u32>,
}

impl TurkishTokenizer {
    fn new() -> Self {
        let roots = Self::load_json("kokler_v04.json");
        let suffixes = Self::load_json("ekler_v04.json");
        let bpe_tokens = Self::load_json("bpe_v02.json");

        TurkishTokenizer {
            roots,
            suffixes,
            bpe_tokens,
        }
    }

    fn load_json(file_path: &str) -> HashMap<String, u32> {
        let file = File::open(file_path).expect(&format!("Failed to open {}", file_path));
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).expect(&format!("Failed to parse {}", file_path))
    }

    fn is_uppercase(c: char) -> bool {
        c.is_uppercase()
    }

    fn split_by_uppercase(word: &str) -> Vec<(String, bool)> {
        let mut result = Vec::new();
        let mut current_part = String::new();
        let mut is_first_char = true;
        
        for c in word.chars() {
            if Self::is_uppercase(c) && !is_first_char {
                if !current_part.is_empty() {
                    result.push((current_part, false));
                }
                current_part = String::new();
                current_part.push(c.to_lowercase().next().unwrap());
                result.push((current_part.clone(), true));
                current_part = String::new();
            } else {
                current_part.push(c.to_lowercase().next().unwrap());
            }
            is_first_char = false;
        }
        
        if !current_part.is_empty() {
            result.push((current_part, false));
        }
        
        // Handle first character
        if !result.is_empty() && Self::is_uppercase(word.chars().next().unwrap()) {
            result[0].1 = true;
        }
        
        result
    }

    fn match_root(&self, word: &str) -> Option<(String, u32)> {
        // Try exact match first
        if let Some(&id) = self.roots.get(word) {
            return Some((word.to_string(), id));
        }

        // Try trimming from the end, respecting UTF-8 boundaries
        let mut chars: Vec<char> = word.chars().collect();
        while chars.len() > 1 {
            chars.pop(); // Remove last character
            let current = chars.iter().collect::<String>();
            if let Some(&id) = self.roots.get(&current) {
                return Some((current, id));
            }
        }

        None
    }

    fn match_suffix(&self, suffix: &str) -> Vec<(String, u32)> {
        let mut result = Vec::new();
        let chars: Vec<char> = suffix.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let mut found = false;
            for end in (start + 1..=chars.len()).rev() {
                let substr: String = chars[start..end].iter().collect();
                if let Some(&id) = self.suffixes.get(&substr) {
                    result.push((substr, id));
                    start = end;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }

        if start < chars.len() {
            // If there are remaining characters, use BPE
            let remaining: String = chars[start..].iter().collect();
            let bpe_tokens = self.match_bpe(&remaining);
            result.extend(bpe_tokens);
        }

        result
    }

    fn match_bpe(&self, word: &str) -> Vec<(String, u32)> {
        let mut result = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let mut found = false;
            for end in (start + 1..=chars.len()).rev() {
                let substr: String = chars[start..end].iter().collect();
                if let Some(&id) = self.bpe_tokens.get(&substr) {
                    result.push((substr, id));
                    start = end;
                    found = true;
                    break;
                }
            }
            if !found {
                // If no match found, take the first character as a token
                let c: String = chars[start..start + 1].iter().collect();
                if let Some(&id) = self.bpe_tokens.get(&c) {
                    result.push((c, id));
                }
                start += 1;
            }
        }
        result
    }

    fn tokenize(&self, text: &str) -> TokenizerOutput {
        let mut tokens = Vec::new();
        let mut ids = Vec::new();

        for word in text.split_whitespace() {
            let mut word_tokens = Vec::new();
            let mut word_ids = Vec::new();

            // Split by uppercase if necessary
            let parts = if word.chars().any(Self::is_uppercase) {
                let split_parts = Self::split_by_uppercase(word);
                for (part, is_uppercase) in split_parts {
                    if is_uppercase {
                        word_tokens.push("<UPCL>".to_string());
                        word_ids.push(self.roots.get("<UPCL>").copied().unwrap_or(0));
                    }
                    if !part.is_empty() {
                        // Try to match as root
                        if let Some((root, root_id)) = self.match_root(&part) {
                            let root_len = root.len();
                            word_tokens.push(root);
                            word_ids.push(root_id);

                            // Check for suffixes
                            let remaining = &part[root_len..];
                            if !remaining.is_empty() {
                                let suffix_tokens = self.match_suffix(remaining);
                                for (token, id) in suffix_tokens {
                                    word_tokens.push(token);
                                    word_ids.push(id);
                                }
                            }
                        } else {
                            // Use BPE as fallback
                            let bpe_tokens = self.match_bpe(&part);
                            for (token, id) in bpe_tokens {
                                word_tokens.push(token);
                                word_ids.push(id);
                            }
                        }
                    }
                }
                Vec::new() // Parts are already processed
            } else {
                vec![word.to_string()]
            };

            // Process non-uppercase parts
            for part in parts {
                // Try to match as root
                if let Some((root, root_id)) = self.match_root(&part) {
                    let root_len = root.len();
                    word_tokens.push(root);
                    word_ids.push(root_id);

                    // Check for suffixes
                    let remaining = &part[root_len..];
                    if !remaining.is_empty() {
                        let suffix_tokens = self.match_suffix(remaining);
                        for (token, id) in suffix_tokens {
                            word_tokens.push(token);
                            word_ids.push(id);
                        }
                    }
                } else {
                    // Use BPE as fallback
                    let bpe_tokens = self.match_bpe(&part);
                    for (token, id) in bpe_tokens {
                        word_tokens.push(token);
                        word_ids.push(id);
                    }
                }
            }

            tokens.extend(word_tokens);
            ids.extend(word_ids);
        }

        TokenizerOutput { tokens, ids }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_text>", args[0]);
        std::process::exit(1);
    }
    
    let tokenizer = TurkishTokenizer::new();
    let input = &args[1..].join(" "); // Join all arguments after program name
    
    let output = tokenizer.tokenize(input);
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
