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

        // Try trimming from the end
        let mut current = word.to_string();
        while current.len() > 1 {
            if let Some(last_char_boundary) = current.char_indices().last().map(|(i, _)| i) {
                current.truncate(last_char_boundary);
                if let Some(&id) = self.roots.get(&current) {
                    return Some((current, id));
                }
            }
        }

        None
    }

    fn match_suffix(&self, suffix: &str) -> Vec<(String, u32)> {
        let mut result = Vec::new();
        let mut remaining = suffix.to_string();

        while !remaining.is_empty() {
            let mut found = false;
            for i in (1..=remaining.len()).rev() {
                let substr = &remaining[..i];
                if let Some(&id) = self.suffixes.get(substr) {
                    result.push((substr.to_string(), id));
                    remaining = remaining[i..].to_string();
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }

        if !remaining.is_empty() {
            // If there are remaining characters, use BPE
            let bpe_tokens = self.match_bpe(&remaining);
            result.extend(bpe_tokens);
        }

        result
    }

    fn match_bpe(&self, word: &str) -> Vec<(String, u32)> {
        let mut result = Vec::new();
        let mut remaining = word.to_string();

        while !remaining.is_empty() {
            let mut found = false;
            for i in (1..=remaining.len()).rev() {
                let substr = &remaining[..i];
                if let Some(&id) = self.bpe_tokens.get(substr) {
                    result.push((substr.to_string(), id));
                    remaining = remaining[i..].to_string();
                    found = true;
                    break;
                }
            }
            if !found {
                // If no match found, take the first character as a token
                let c = remaining.chars().next().unwrap().to_string();
                let c_len = c.len();
                if let Some(&id) = self.bpe_tokens.get(&c) {
                    result.push((c.clone(), id));
                }
                remaining = remaining[c_len..].to_string();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenization() {
        let tokenizer = TurkishTokenizer::new();
        let input = "Kitabı ve defterleri";
        let output = tokenizer.tokenize(input);
        
        // Test token presence
        assert!(output.tokens.contains(&"<UPCL>".to_string()));
        assert!(output.tokens.contains(&"kitab".to_string()));
        assert!(output.tokens.contains(&"ı".to_string()));
        assert!(output.tokens.contains(&"ve".to_string()));
        assert!(output.tokens.contains(&"defter".to_string()));
        assert!(output.tokens.contains(&"ler".to_string()));
        assert!(output.tokens.contains(&"i".to_string()));

        // Test token sequence
        assert_eq!(output.tokens[0], "<UPCL>");
        assert_eq!(output.tokens[1], "kitab");
        assert_eq!(output.tokens[2], "ı");
        assert_eq!(output.tokens[3], "ve");
        assert_eq!(output.tokens[4], "defter");
        assert_eq!(output.tokens[5], "ler");
        assert_eq!(output.tokens[6], "i");

        // Test IDs
        assert_eq!(output.ids[0], 20009); // <start_of_turn>
        assert_eq!(output.ids[1], 22201); // a
        assert_eq!(output.ids[2], 22202); // b
        assert_eq!(output.ids[3], 22203); // c
        assert_eq!(output.ids[4], 22204); // d
        assert_eq!(output.ids[5], 22205); // ed
        assert_eq!(output.ids[6], 22206); // you
        assert_eq!(output.ids[7], 20010); // <end_of_turn>
        assert_eq!(output.ids[8], 20005); // <bos>
    }
}

