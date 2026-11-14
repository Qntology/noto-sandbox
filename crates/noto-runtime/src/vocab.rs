//! The texts the SigLIP2 token table is primed with (`noto_ml::Siglip2::prime_vocab`): what the engine embeds at
//! start-up and the endings commands end with, so the rows a session needs are on the device before its first phrase.
//! Any other text (a free-form command) still embeds: its rows come from the mapped checkpoint the first time.

use anyhow::Result;
use noto_search::{Lexicon, VisionPrompting};
use noto_vision::locate::apply_template;
use serde_json::Value;

/// Every string of the world lexicon (names, aliases, phrases, quantities), the prompt templates applied to the
/// lexicon's vision phrases and the background classes, the logistics dictionary's prompts and labels, and the Korean
/// command endings (`noto_search::tails`). Sorted, without duplicates.
pub fn vocab_texts(lexicon_json: &str, dictionary: Option<&noto_ml::Dictionary>) -> Result<Vec<String>> {
    let mut out = Vec::new();
    collect_strings(&serde_json::from_str::<Value>(lexicon_json)?, &mut out);
    let lex = Lexicon::from_json(lexicon_json)?;
    let prompting = VisionPrompting::from_lexicon_json(lexicon_json)?;
    let phrases: Vec<String> = lex
        .vision
        .iter()
        .flat_map(|e| std::iter::once(e.semantic.clone()).chain(e.bias.iter().map(|p| p.text.clone())).chain(e.prejudice.iter().map(|p| p.text.clone())))
        .chain(prompting.background.iter().cloned())
        .collect();
    for t in &prompting.templates {
        for p in &phrases {
            out.push(apply_template(t, p));
        }
    }
    if let Some(d) = dictionary {
        for e in &d.entries {
            out.push(d.prompt(e));
            out.push(e.label_en.clone());
            out.push(e.label_ko.clone());
        }
    }
    out.extend(noto_search::tails::VERB_TAILS.iter().filter(|t| !t.is_empty()).map(|t| t.to_string()));
    out.retain(|s| !s.trim().is_empty());
    out.sort();
    out.dedup();
    Ok(out)
}

fn collect_strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(a) => a.iter().for_each(|x| collect_strings(x, out)),
        Value::Object(m) => m.values().for_each(|x| collect_strings(x, out)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEX: &str = include_str!("../../../data/world_lexicon.json");
    const DICT: &str = include_str!("../../../data/logistics_dictionary.json");

    #[test]
    fn the_priming_texts_cover_what_the_engine_embeds() {
        let dict = noto_ml::Dictionary::from_json(DICT).unwrap();
        let texts = vocab_texts(LEX, Some(&dict)).unwrap();
        let has = |s: &str| texts.binary_search(&s.to_string()).is_ok();
        let lex = Lexicon::from_json(LEX).unwrap();
        let prompting = VisionPrompting::from_lexicon_json(LEX).unwrap();
        for e in &lex.vision {
            assert!(has(&e.semantic), "{}", e.semantic);
            for t in &prompting.templates {
                assert!(has(&apply_template(t, &e.semantic)), "{t} × {}", e.semantic);
            }
        }
        for b in &prompting.background {
            assert!(has(b), "{b}");
        }
        for e in &dict.entries {
            assert!(has(&dict.prompt(e)) && has(&e.label_ko), "{}", e.id);
        }
        assert!(has("해줘") && has("주세요"), "command endings");
        assert!(texts.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
    }
}
