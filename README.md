# RustPotion
*Oxidized [Tokenlearn](https://github.com/MinishLab/tokenlearn) for blazingly fast word embeddings.*
 

## Example

```
cargo add rustpotion 
```

```rust
use rustpotion::{RustPotion, PotionModel};
use std::path::Path;

fn main() {
    let model = RustPotion::new(PotionModel::BASE2M, Path::new("models"));
    model.encode("test");
}
```

## Why
\> be me

\> saw cool project

\> said "why not rust"

\> Now cool project is in rust

## Speed

Because tokenlearn is so blazingly fast (mainly cause it's only just an average of some word vectors), the limiting factor is actually the tokenizer implementation.

That's why it's good news that we get ~27MB/s of input sentences for potion-base-2M, which on par (if not marginally better) with most other [high performing tokenizers](https://github.com/huggingface/tokenizers).

## Performance

This project is just a rustified version of Tokenlearn so all the results (should be) the same.

| Name | MTEB Score |
| --- | --- |
| potion-base-8M | 50.03 |
| potion-base-4M | 48.23 |
| potion-base-2M | 44.77 |

## Limitations

1. Only english; unicode slicing is pain
2. No python bindings, just use [Tokenlearn](https://github.com/MinishLab/tokenlearn) (it's secretly rust if you look deep enough)
3. **RustPotion::encode_many** is multithreaded and will use all available resources
4. No limit on sentence length, but performance starts to dip after 500 tokens (~100 words) so be careful.


## Final thoughts
Don't use in production unless you like living on the edge