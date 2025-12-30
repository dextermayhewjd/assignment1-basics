strings = [
    "hello",
    # "こんにちは",   # 日文
    # "hello こんにちは",
    # "😀",            # emoji
]

# encodings = ["utf-8", "utf-16", "utf-32"]
encodings = ["utf-8"]

for s in strings:
    print(f"\nString: {repr(s)}")
    print(f"Number of characters: {len(s)}")

    for enc in encodings:
        b = s.encode(enc)
        print(f"  {enc}:")
        print(f"    bytes: {b}")
        print(f"    byte values: {list(b)}")
        print(f"    number of bytes: {len(b)}")

