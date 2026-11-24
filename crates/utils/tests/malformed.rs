use ferment_utils::id;

#[test]
fn no_string_input_can_panic_the_module() {
    let nasty = [
        "",
        " ",
        "\0",
        "\n",
        "é",
        "🎛",
        "日本語",
        "\u{200b}",
        "outro",
        "OUTRO",
        "ferment-outro",
        "-",
        "--",
        "+0000000000000000",
        "01983265-89db-7d2a-8b0b-3ce68b2a5cee",
        "01k0s6b2evfmn8p2swwt5jmq7e",
        "01k0s6b2evfmn8p2swwt5jmq7é", // 26 chars, one multi-byte
        "ééééééééééééééééééééééééé é",
        &"z".repeat(25),
        &"z".repeat(26),
        &"z".repeat(27),
        &"z".repeat(1000),
        "\u{10FFFF}",
        "01k0s6b2evfmn8p2swwt5jmq7e\0",
    ];
    let candidates: Vec<String> = ["01m0qwrt88f5v8r0abm12yrtys".to_owned()].into();
    for input in nasty {
        let _ = id::is_valid(input);
        let _ = id::compact(input);
        let _ = id::to_uuid(input);
        let _ = id::of_uuid(input);
        let _ = id::timestamp_ms(input);
        let _ = id::is_short_shaped(input);
        let _ = id::resolve(input, &candidates);
        let _ = id::short_seeded(input);
    }
}

/// `compact` returns a borrow into its argument; a multi-byte tail must not be
/// sliced through a character boundary.
#[test]
fn compact_never_splits_a_character() {
    for input in [
        "01k0s6b2evfmn8p2swwt5jmq7é",
        "ééééééééééééééééééééééééééé",
        "🎛🎛🎛",
    ] {
        assert_eq!(
            id::compact(input),
            input,
            "non-convention input must pass through whole"
        );
    }
}
