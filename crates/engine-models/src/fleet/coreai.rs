//! Qualified macOS AOT pins from ferment-models-export, 2026-11-15.
//! Select one architecture before manifest validation; never install all chips.
//! These immutable constants have no runtime dependency on exporter files.
use super::{
    ArtifactSource, BackendKind, MODEL_BASE_URL, ModelArtifact, ModelId, ModelManifest, Placement,
};

struct Pin {
    architecture: &'static str,
    tree: &'static str,
    manifest: &'static str,
}
struct Bundle {
    model: ModelId,
    stem: &'static str,
    pins: &'static [Pin],
}

pub(super) fn append(manifest: &mut ModelManifest, architecture: &str) {
    for bundle in BUNDLES {
        let Some(pin) = bundle
            .pins
            .iter()
            .find(|pin| pin.architecture == architecture)
        else {
            continue;
        };
        let definition = manifest
            .models
            .iter_mut()
            .find(|model| model.id == bundle.model.name())
            .expect("Core AI model must be in the base fleet");
        let baseline = definition
            .artifact_for(BackendKind::CoreMl)
            .expect("Core ML fallback");
        let file_name = format!("{}.{}.aimodelc", bundle.stem, architecture);
        let artifact = ModelArtifact {
            backend: BackendKind::CoreAi,
            source: ArtifactSource::ManifestPrefix {
                prefix_url: format!("{MODEL_BASE_URL}/{file_name}"),
                manifest_sha256: pin.manifest.into(),
            },
            file_name,
            sha256: Some(pin.tree.into()),
            license: baseline.license.clone(),
            source_url: baseline.source_url.clone(),
            placement: Some(Placement::CoreAi),
        };
        definition.artifacts.push(artifact);
    }
}

const BUNDLES: &[Bundle] = &[
    Bundle {
        model: ModelId::GlapAudio,
        stem: "glap_audio_mel_fp16",
        pins: &[
            Pin {
                architecture: "macos.h13c",
                tree: "cc5c683e646313f6aa86cf3af381afaf6220d3023c76fb27bc66fccd7665be88",
                manifest: "87483fef355949036c11c0334e1344b8c9d101deeb3ae9d5449411ae76130a84",
            },
            Pin {
                architecture: "macos.h13d",
                tree: "9a80a7c7a1e8e83d3a3a67c3a3c569cbeeb87421e822c6e817c9436c7c8f85bf",
                manifest: "acd0b950b3cf3639629a75ad5078121efd309f0f534f55c458d64a1b9fd12bf2",
            },
            Pin {
                architecture: "macos.h13g",
                tree: "9f63f3609366bbb48fbb6aef49591c09a73e30dd86d3c035583e1196ee0cb90b",
                manifest: "ab5fe1013adc3356ab138ff4e9ad6cb5be65eea8dd158d2cbd6da3fac68899c0",
            },
            Pin {
                architecture: "macos.h13s",
                tree: "af6f6b7116a130d70f791a86b4c67a8985b13facae84dda5bff90971bd03b1d4",
                manifest: "73bf659a65329d70f3e44b13384994791dd7b94e9e2b6b8535b9bf818d0c7809",
            },
            Pin {
                architecture: "macos.h14c",
                tree: "67291f9947f5fa50ea27b047908ad19d32ecc876bbce8586c4c1cc693bcade27",
                manifest: "bf39441ce9520920bb15cf58828db7a709c6ad35c5f5133d829abcf0b0ae15e3",
            },
            Pin {
                architecture: "macos.h14d",
                tree: "9d57e6e5dc5d4dee3dadc21c0b4abc00f39545515238b400106d0fdd1d13c0d0",
                manifest: "d4329ef70a1ac70d78afe0c7705d851b22a24c49aa638f278d068a5cf702ec32",
            },
            Pin {
                architecture: "macos.h14g",
                tree: "711f85d35f44d91165c87b9527da63f69b0188ae78ca61291b50eadbf9d878b3",
                manifest: "2e62c8938a3d7770dbe993191293ba78eb8a10aac3dedd3d8119ee57703dc043",
            },
            Pin {
                architecture: "macos.h14s",
                tree: "b66996672d77cdcda4ec0bf7ef1489f37813fe97c60f07ce19b423e6d88aac4e",
                manifest: "f41f23c12b56c886385db22b6b19bc14bcb50739e2e839be622fb0ef769a6153",
            },
            Pin {
                architecture: "macos.h15c",
                tree: "38d848d2ca964258d6011eff6d40c192e54a15da4cd17a8d5ff59f7f9a8989a7",
                manifest: "6a22bc5a3c28b4f68b0020a33bbe9a56602e9ccbdd461529c4609a28f525ed86",
            },
            Pin {
                architecture: "macos.h15d",
                tree: "aa6e11b74d4d03d0cdc935f9e55af6b068bd6e097e1297e0e30d77bca405c4a9",
                manifest: "8e80e1681dcb7ceaa62ea1b9688fdd267df4cf5613f8c1b99695f122d672aa35",
            },
            Pin {
                architecture: "macos.h15g",
                tree: "ef90cf605f7a773cd064967d7ccd4e444424484554970264cb18f4060ccbc630",
                manifest: "fc25df0427fc4c4bcc6967ec8dacdb4c3fa45c9a487c06400d91e62a8dd8d0c9",
            },
            Pin {
                architecture: "macos.h15m",
                tree: "4d92ca952275458f23911bc965a90f8f588f717b70d39df34c758e868ce5021c",
                manifest: "03012aa442d72a846bc3a0fe7b89f75a9ad956bb037ba011f3f419d61c60fbe3",
            },
            Pin {
                architecture: "macos.h15s",
                tree: "bec2c2f4d0d0cac15c2fb6c1964d834d220281e1572374a880f114a978c07b10",
                manifest: "1d5325aa7b2ccd117a165a53671bc3d4477c5713aacf9edf633528186be42779",
            },
            Pin {
                architecture: "macos.h16c",
                tree: "06bd760b4ebde29260c9dde4ec18a616d7bb347708ade8ab0910e27694d3c45c",
                manifest: "9c0c1c685ca3c30634c5191a2f3b313cdb145aa0081280031e2e493d858ab394",
            },
            Pin {
                architecture: "macos.h16g",
                tree: "2e15010cc252cee36b64beb5459a26dd056e8c6885f820682ffb092bbcb92953",
                manifest: "362c1494838a68e11e2b84634764423b452a10269295e91d339dca7a5efe3780",
            },
            Pin {
                architecture: "macos.h16s",
                tree: "8b688bc5ef4cd7c55cd643f75ca530ac3f4321b8807fce531a4846469210089f",
                manifest: "eeb7bd8312227e00d851faa455a57b25c8edf3494f38cdb39eb6904ff77a38a1",
            },
            Pin {
                architecture: "macos.h17c",
                tree: "c9756af5d02363b44ce6c0ec6bc3c7bb8e6cb7555c1ab152f81e35efaae73ef1",
                manifest: "787419fe939fce8fc46c54a8e2dfc6ef51677327a06890f39263def1f9d726ae",
            },
            Pin {
                architecture: "macos.h17d",
                tree: "969012f92e489ee42e7cb78f92550120050732f6bb26f52bcfbe4788fd457420",
                manifest: "75eca16e8323def1a42e2201b9c97a3897772af6c73203251facb6c9dd6ee601",
            },
            Pin {
                architecture: "macos.h17g",
                tree: "39a9711ed9a120370b92a256e511c8bcbd3e9b83a2be3b37ff52ea53abc1ad98",
                manifest: "aeea48ee7c37866f21f490931e59e677219cb80711d379a28921176ccb4c2fcd",
            },
            Pin {
                architecture: "macos.h17p",
                tree: "ddd2a7888d1bd0c586235fcb84a9cf4d2fc022e69d48eb4b797938d8dc20838c",
                manifest: "4373bda0a2aa9b95e49573db34e5d74c1014d98e1aedb8cb31aa4e048cde2d0e",
            },
            Pin {
                architecture: "macos.h17s",
                tree: "c5370c138928350c65b4037f6ac3857f0dcab349fe0c43ea7dbf40fe1b13dd5d",
                manifest: "d2068d45a730dbbe531c5b45621091c8af7d2bd961d19a5a59904d48b0c1cae8",
            },
            Pin {
                architecture: "macos.h18g",
                tree: "d351c796648de7f8686304cab60156933c29413361338287e65d4a7ffdbe7f4e",
                manifest: "b40fb528d36deea73e3c78891e1ddebe5934c298e929ba31ab605139ec6868a3",
            },
        ],
    },
    Bundle {
        model: ModelId::GlapText,
        stem: "glap_text_fp16",
        pins: &[
            Pin {
                architecture: "macos.h13c",
                tree: "f138992ef5138f57955cc89d500dff32a569d51b0b09cd73d73abe4a0bdab27d",
                manifest: "46b698add91c3160a30c2c7dea06eb15c45d8b721aab562ed1ece6ca3f0d786f",
            },
            Pin {
                architecture: "macos.h13d",
                tree: "8d5aede6e333129ae5b2e3b9ecf67673fa9701c2bfdcc7bf5b98cb3591244b9f",
                manifest: "f56a8c8abae9f8c88e898c06f63c382ec6639591254aba9465da5c0f50151232",
            },
            Pin {
                architecture: "macos.h13g",
                tree: "9818a95d41a05001a8c57161ea3351d5b7053ebea2f172355f294963dea41258",
                manifest: "f5df758458253ace0f1fad876edd760fb641568c38e6eb8ef56c8763cb1fb18f",
            },
            Pin {
                architecture: "macos.h13s",
                tree: "0bc920c76b9a1bda01ab5eb0e97c4e10ec5386aeac96f452479fed824dda475e",
                manifest: "35638c010282fe2274e44d45009c32cd1b250f8ab6b1e6bc58e354c344f56c79",
            },
            Pin {
                architecture: "macos.h14c",
                tree: "b84cf74241bae39188d5570de72fdbb6920da6b898c435536714bbcda0a9b83a",
                manifest: "365354468eb57afb07dbf9f1c911e5bf39c0a357013c12d9454aa0461736b703",
            },
            Pin {
                architecture: "macos.h14d",
                tree: "1ee61d18e8f8da7000e61c524acdb90c10892cb7a0d7ec4e2f479ebba1494229",
                manifest: "c313e1b0bf83de84fc804cc98d56b7cd7694890e1ba8ed6e6c80e647719c457c",
            },
            Pin {
                architecture: "macos.h14g",
                tree: "308e72409d2d43ef9baf61a62fdd9d21bcb32e5dd08d20c4c98c05231cc06af4",
                manifest: "0938e89bdf49b1a61e83a6204bd22457294718d86dde2a48eecc410281a797c8",
            },
            Pin {
                architecture: "macos.h14s",
                tree: "3d33ed6a665ac462c342b6577183e2af3beb87121f0d42396a9fbbe5ae47790d",
                manifest: "04bce738db536c0240c0b08ed2230a6e4095ab25aa1d265b985e04b365b3a81f",
            },
            Pin {
                architecture: "macos.h15c",
                tree: "ef0c354fbd3934eb33f15197b181a050facc77cd4eedf78411e41171750b84ac",
                manifest: "e839340cd433ed7e8c7b6149f888e5b1b7a7af08410d17c6640fab6a0e44b1bf",
            },
            Pin {
                architecture: "macos.h15d",
                tree: "a1edfb610a5f3c1fa5c1cca76c832e4c54b525a41a3bd9eb6367deb45a00693a",
                manifest: "e4f3128208ae460af3e9043f05b02132a2ff8133c90c8f7481e6345e42ca3f2d",
            },
            Pin {
                architecture: "macos.h15g",
                tree: "255c7facc84b10ff26887f01cfde66bb77d187987393eaa04b71c835a90eb8ae",
                manifest: "9842707f2b4fd6bc834fbd20a1945212dbec5383042be0b0b89e7aac65f385dd",
            },
            Pin {
                architecture: "macos.h15m",
                tree: "647e89278e9807be1ccd262cdc74b7dd68a7af88f91e2ac24c04b6a227164009",
                manifest: "e298969c258eb365d9c1a8131f23ccbf0e3370d571f7837f7a78a79256fb2bdc",
            },
            Pin {
                architecture: "macos.h15s",
                tree: "869c4921520f684afbfedcd2dd5c4c8110fe48f862ee579b093d799ad9e14359",
                manifest: "194aff320a88a0ffd2fdffb207189a336200acadc0cfa6bb19cc087c77b925c2",
            },
            Pin {
                architecture: "macos.h16c",
                tree: "8ebadefaac1edfc0fea90860fbd6f64c215246cac5ca624e8a99244876ecf26c",
                manifest: "35fad1ef6491c3a21ff454ff8c833cdc36b0b033f4d7ff244ee71c6a57429124",
            },
            Pin {
                architecture: "macos.h16g",
                tree: "4c26cbc6314b61bcceaf7dc461edc9ccae7df191656a21389b06e80519ab1063",
                manifest: "6b762f5472fd058aed7c1cdd722f0ad86ec55ee2c7549c7def1d341f23b6d0db",
            },
            Pin {
                architecture: "macos.h16s",
                tree: "1107db0b01736532e7d56041096f34770a56abd3394b928fb5aa1b56f49e803b",
                manifest: "7be40e3ea3cd8a91cdedef63974745903a07a8a93b71f637552c503e780a8aa4",
            },
            Pin {
                architecture: "macos.h17c",
                tree: "66c01a17475cb089593134bd13f53428bc295f8cc133cb0a8513ea0aee7e8a06",
                manifest: "7da9a9c3e1db37b379b84eda452675b823ff58c7f85e8c312cd9b3e4d1bf8382",
            },
            Pin {
                architecture: "macos.h17d",
                tree: "9ff7dbc9bf9ef85e49712c24703975ed9eff183b51b151da66b2c304d3985a6d",
                manifest: "0eb5b36605939f11a8272345a4759ecb3c2a9f91e064188a926ae2d6a034cdaa",
            },
            Pin {
                architecture: "macos.h17g",
                tree: "0c245c0ff6e3f79ec69a2f873219e4d33eeffe1f0c1e4b1cfb5f6c793b8b2ac1",
                manifest: "dd45a551def6652372565e9b787bb1a0059546a17045556924e2a6269ccafa91",
            },
            Pin {
                architecture: "macos.h17p",
                tree: "9da178ec0ee0ee8a2d2e2f9f24a3f7aa25d6f24af2c8efe07dce0f0b62f90d9c",
                manifest: "47beb313c4b787eac7f708ba5f0de4857a7a50487b50e0d83d5de483a93d72e8",
            },
            Pin {
                architecture: "macos.h17s",
                tree: "6a6892526770af18584b8e6256115c2dfe494fb0db08d8c282dfb9aa3201334a",
                manifest: "aa7838dcaab0ba90ad151cccee35af51ab6f2f9e317a2ee1cbd6f0ae5c5011ac",
            },
            Pin {
                architecture: "macos.h18g",
                tree: "6a67f9ed0ee306924a49382c55b634506dcc82f05548445ef9a4e3ae19eef99e",
                manifest: "2a977283d4d8ca6e78782810692f7927486b7c806fc3ce3591043aa92face86f",
            },
        ],
    },
    Bundle {
        model: ModelId::Siglip2Image,
        stem: "siglip2_base_224_image_fp16",
        pins: &[
            Pin {
                architecture: "macos.h13c",
                tree: "4badd4b6efc7e95801dd1ec2a07eaf21c9834a573bad0a163ce6ec8ddb096f7e",
                manifest: "8f6d1745288aa92d797b42ce3678159354cf34f5a62b9dc1eb2c423b0e1c857d",
            },
            Pin {
                architecture: "macos.h13d",
                tree: "61e350249ca362086e86dd5ea90b2ff1304a1a31a329fd2c2de03f1343f2bbea",
                manifest: "50f0ecc761eaa1be6dda514b01492ac90440525be88127a71e8bc0b11f7c6274",
            },
            Pin {
                architecture: "macos.h13g",
                tree: "b2cf008c4e17f385ee26a0b40c0af0d5bada519d356f6f77cf4512fd2e24642e",
                manifest: "f12ff738f49806ce4b1f1c6513b693c9c8def379c8f01d0f5c5be8e5ddcb89e9",
            },
            Pin {
                architecture: "macos.h13s",
                tree: "d4d432a842402bd702e6eca66333c19abd7a9d5a162b72a1590d184cee6bf9b2",
                manifest: "7a08ffec7af89810d55b6039f9d4c965366316171bc90416e4ee8ae81c79dfae",
            },
            Pin {
                architecture: "macos.h14c",
                tree: "aa912ed9826a8eabca6cd1b8ab56a1c31fc5a5cd45da4b9924ea442392423ed4",
                manifest: "2a0013405db0ddf0cf34fc7e4a732816b757c4856fe94d6ae5581d0c767fdc8c",
            },
            Pin {
                architecture: "macos.h14d",
                tree: "913d15cb4032666f5981ecc9702a651068f3e8092d41f9eb1d56eda11c0acb9e",
                manifest: "4920f71a814c228eb26c16810fb4878ea63d55ee0918db1753676ac5a894d3b7",
            },
            Pin {
                architecture: "macos.h14g",
                tree: "b5cd2e206b16c7083329b7268508b6e4b63c49a5d395ad8e22a625f95e516a95",
                manifest: "be751cdab709eee5d310b680cbcdb81cc77dfec6924dab2a42e64588a8b354fe",
            },
            Pin {
                architecture: "macos.h14s",
                tree: "b93df9bf0e419a4e10dc72677ca65629c94ad9a5daf9658a99e664d36ee0c4fd",
                manifest: "d72e22b8a0f0461a2d4edf1751c5817c54b066b82d6e9ded61be3ef0ac57faca",
            },
            Pin {
                architecture: "macos.h15c",
                tree: "bc7970da630832f9db5b101553dfbf658dff8c77dcdc39a1b58e9c5ba28e6962",
                manifest: "8df0a5f8cca576fbc70e574521a2a6a58fa685a21c047d0230519242348c4b48",
            },
            Pin {
                architecture: "macos.h15d",
                tree: "8d037393cc68a516fc301e7b1e13baa8365427eaa8f611394c6c59e0302c32b3",
                manifest: "da52a80750ac642c0c3e1b42fdc44784be824f98bc79a7ecc87689851cd5cdd3",
            },
            Pin {
                architecture: "macos.h15g",
                tree: "18c08ef6ecd8f52689845ec1f61e8468966e43d7c2d9a2537977a97766f4ab90",
                manifest: "94d5ccef3c7fe59bf7a2244c8b2813df112e090d22e07468a2625fd78754e990",
            },
            Pin {
                architecture: "macos.h15m",
                tree: "c17229e0d9cefce97c6c0ed44703229695769abe2abb1065d2471eab7321666f",
                manifest: "a4c74db0b9f703122342dcde065d624d410c58496409aa3ba48fd57dbdae0a05",
            },
            Pin {
                architecture: "macos.h15s",
                tree: "75be665cb3b7a6a175ea016369cad67480a5a71150e8ece2ff355ed4c164dc2f",
                manifest: "d5aa3a204aa40c67e842ac1c763a62e8a10ffa9a04fd9d19cadffd6d150c54a3",
            },
            Pin {
                architecture: "macos.h16c",
                tree: "fc07c20c45ba3cfdfc246f6789f4cc364ca9cf0c5f7b339104c2b56cbc7831c7",
                manifest: "2ca6dec323ebfce8d3da01e33c54c87f804e4c7dbd9d675778b51be4f70d5405",
            },
            Pin {
                architecture: "macos.h16g",
                tree: "64cea2d48a658c25c883fa0923523c221ca5de0de653cf0560dddd2743949531",
                manifest: "e6b6de04cc8ae53047ae8600778fdfb7c5f4cdf3ea524378951edb79435464ea",
            },
            Pin {
                architecture: "macos.h16s",
                tree: "2f5d134d124d216f81302d0f5a7a7cc4ae51ec3f4b32935cae01442388d47510",
                manifest: "c3ec504e4f4f02f8b9c823f837f45f389534e8a6fd2009cae7b75fab525f49b1",
            },
            Pin {
                architecture: "macos.h17c",
                tree: "faca38600d9e416ed1fc375609b573db6dec571c8171c320568c183efbf4e7eb",
                manifest: "a9933d6abcd8d94d0feeea8e3f02c923c7825ae0f760b31ceb26b03688073a8c",
            },
            Pin {
                architecture: "macos.h17d",
                tree: "8732f6a04775bf87c7be34c020803590716a8aa22fc330f85dbca359a732648d",
                manifest: "1e28baedf74d4696390907477efd9fdb5c5bb81586f7930743f04f123433f212",
            },
            Pin {
                architecture: "macos.h17g",
                tree: "be4213864c23bb73b5a0e3804a4712b5b767db0f3c272dd9efb17e70058ad852",
                manifest: "00973abb456ec5997ffb4f9b4157e7f8269d939d48dc99b8f0a70e70075dba6a",
            },
            Pin {
                architecture: "macos.h17p",
                tree: "05d1b081ce91e203feac640c89ba393eba1603236bb9f80aceb1503436e85115",
                manifest: "f557969647cde3687eef0f7f9e9f954ad265102ec78e0211e610369336c8fa6c",
            },
            Pin {
                architecture: "macos.h17s",
                tree: "2f5a6cd849aa0fc4726c578860ac9831915b81c500039153a06ee882eeae41b5",
                manifest: "91f50a16d72e1db1ce76efb770cd6017711d3b7940bbfaea3987699ddc7ce9fd",
            },
            Pin {
                architecture: "macos.h18g",
                tree: "995c52b324e63847b38260f55fd162db76111572adc43ec9495bd615d5343eeb",
                manifest: "9fae37896778df20eabb811a5532ae0bae1564c534f6442468f9bbc8d25466a0",
            },
        ],
    },
    Bundle {
        model: ModelId::Siglip2Text,
        stem: "siglip2_base_224_text_fp16",
        pins: &[
            Pin {
                architecture: "macos.h13c",
                tree: "9901eb2c64c092cf3b1d1e8ca1b0d8c57282b6ea99e2a94c1c221fd0c5ade99c",
                manifest: "cf2fab804e0c7e68d139dfc1a6839dfec63bba5db88accb5b024573a5b2ef7d8",
            },
            Pin {
                architecture: "macos.h13d",
                tree: "0575df43c7aca6f47fea6b671e4f340214a1e54b72b1f82c783cacca12ddb0e1",
                manifest: "91cf2b0589b2f134793098f63093816dda29bb8819246520b47cec89b872fb35",
            },
            Pin {
                architecture: "macos.h13g",
                tree: "0c5050651dae07406fb8aa441b3d2aa70a2712ea9d5616a622d1a67f3ffc4dd6",
                manifest: "e53525b5011490bfca661e82449ea470ad53c5b7ab330b6c64d372b439f95822",
            },
            Pin {
                architecture: "macos.h13s",
                tree: "ec90b5903efd558158de8dfe7d4e5146d7223405998b2e4ed81d728746f46999",
                manifest: "d3d2f016ce4bd2b1e863ec4cbf39445e2908de44d1ef33326e13ecaf7067dc32",
            },
            Pin {
                architecture: "macos.h14c",
                tree: "a8e0031cd77b546de076165ffd2df579682f32e80cd4e527047e3a4c98c059e0",
                manifest: "1be8138d5f8a65d2cbde20c75b35894fcb6c4f838c8cd98ba9dc5e77c0981ef0",
            },
            Pin {
                architecture: "macos.h14d",
                tree: "520d9eb189a3646992d45544591316a7e2025f97a1f8170825438516781972f6",
                manifest: "5736f9211d0a349779e5ab099666323063a88021b198b14c7cce4eef72ae1c6e",
            },
            Pin {
                architecture: "macos.h14g",
                tree: "c94de7893be004e246aedd497321c26fd4e76bd6bf348f421b352e9577e2a73d",
                manifest: "e2bf1ab130b298384876f7ae545dc575a7646a679ab1604f15b2facf8659bc2e",
            },
            Pin {
                architecture: "macos.h14s",
                tree: "c16f4fd944e1b607a8728427ebd7de5c660935428db983c661f35c15b64aa506",
                manifest: "49cdd62f68639ebe45b1b1b27dde2661fb2823c8ebb11ac752cb26364865ae24",
            },
            Pin {
                architecture: "macos.h15c",
                tree: "00668d388373bcc225e7b0a0cd0241b6f8a62864cfbc7c6bc9c1796ac11892f6",
                manifest: "9711c48114860b36402a8a761df88d273b1da233313adc5fe11b01d3712f3d82",
            },
            Pin {
                architecture: "macos.h15d",
                tree: "8b77aac7f3b29e617a870373173df890899da9d94b02f687b69c5b1f0ca01288",
                manifest: "38dd109b0273f226fb84025fe853ba99e8409d230b38437843af012796384914",
            },
            Pin {
                architecture: "macos.h15g",
                tree: "5e9a9e86d053fe3dac3314117de1ab1f50cec2cf10fd9bfb6b7e606217dfa58d",
                manifest: "b47e23a4b1cf8a652d6418457071ec1041e6104fb82efc0fe9f45bd0ea464b66",
            },
            Pin {
                architecture: "macos.h15m",
                tree: "ec27550844c3869d5403bfb5cd56320966637074c049dc1499abfbdc772f8e59",
                manifest: "71a27b1091d7237e8183a0b8fe089c9b7d74aed3c6edf6a664f1fc0ec24722de",
            },
            Pin {
                architecture: "macos.h15s",
                tree: "50e344b09c9d86f3017e3f433b2326407f034975a61c9c2e5330b2431e0e5d4a",
                manifest: "481fbb6df1d7593298ae44499e65a1ef9fc938f87993cdc33da7b25295bfd22d",
            },
            Pin {
                architecture: "macos.h16c",
                tree: "06344e5362614bb40a73cc68472119d58410b5c8613a94d693432b05056d733a",
                manifest: "1f16119146484ff593268dafc1bb65e137d205cb8e117e73a4880c8684b062ef",
            },
            Pin {
                architecture: "macos.h16g",
                tree: "d743161bbb2981d4eee4c896e2e3fe663ae707f1848c3dc2100ab0146854029f",
                manifest: "d753c483a861894313db6105a2b95271503d7cbebd9a23dfcadd8c92b0cf084f",
            },
            Pin {
                architecture: "macos.h16s",
                tree: "ea00265fd3dc462667e521a7c9f8bf5a6ecf286061d56f817be113a6fc4c283e",
                manifest: "92ef63f7cde387c2fe060b8264a1a22107eff538909c425d3777bfb5749f04d0",
            },
            Pin {
                architecture: "macos.h17c",
                tree: "b12297b92c4de515df4d6e1f3d06caa53c17a2820158e9eb09b9105e9e519caa",
                manifest: "0f64804ca2ef59239e7f7362f8c9a0e260564088f0bb5fb058ee2996f680c8e1",
            },
            Pin {
                architecture: "macos.h17d",
                tree: "e0891237b9095d58e1e65092881ba3a048fd09cce14e013a54c69737134d01d1",
                manifest: "d2febc9578440d223739e24c9311cb48f7aee2f72b63a2a8325b391fe2905a69",
            },
            Pin {
                architecture: "macos.h17g",
                tree: "bd3bc2ba3355a0b66a3d30fea5224bcc92ce49f24adf5a930a4aabb0d390defd",
                manifest: "631481044f97994728e77063cfd70899b03f8d65e4e3ef82f03b63d63a86c88d",
            },
            Pin {
                architecture: "macos.h17p",
                tree: "0bf5657dbd18dec573ab612f0bd54cc91628b003d7b4c76b72bc5c60c3d71adb",
                manifest: "d178caa9b3062064a5e46dad991c7b09e0b17fbe7eaf28e20fb4824323bee3d8",
            },
            Pin {
                architecture: "macos.h17s",
                tree: "0fc7eca8c9394cd0ed6eac201b289170b7f2bc8d7220d2a1acbb454dabc84e3f",
                manifest: "087782d63406ef7572dfc75a498797753f8291b1410838d09f71d04803cfcf29",
            },
            Pin {
                architecture: "macos.h18g",
                tree: "440356bec934076dc0fd293bd1ae503835cdca95700ee8fb89fed272d636bbd7",
                manifest: "f3d3716b015be68b7c684033bd9434361f74fcd88e77b3ea093d85ee049fce13",
            },
        ],
    },
    Bundle {
        model: ModelId::Mert95M,
        stem: "mert_v1_95m_fp16",
        pins: &[
            Pin {
                architecture: "macos.h13c",
                tree: "abc797aa6869e5a211f74d74645f2a5858a2629630c23c6f3ac1ca90a194ff93",
                manifest: "a7a8687ed89642599a3b0f2bff090fe96b1004d9e288c8e336ac38fa89aa050b",
            },
            Pin {
                architecture: "macos.h13d",
                tree: "40866e4321bc613a1df59e6bef06f1e331c374bf25b1af861583b409dc98c9f1",
                manifest: "7d5c97c4cc8c23b59b27b9e10ee9c5e8610b6dafbd700b018769af29555babaf",
            },
            Pin {
                architecture: "macos.h13g",
                tree: "407c14367d7910da5dbefc56c07f166012ccdc52e36f02caa45006794f32d15d",
                manifest: "0c8f76a3bf9124fb876a1b05b6909caccbf6c7eecc4d577160c9fed7a19aa254",
            },
            Pin {
                architecture: "macos.h13s",
                tree: "50c8145d3206698f6e586bd654f2b144ddbe911933f2b8b768ac1731c29e28b6",
                manifest: "05c1b6dd9b3d74e4bd754e241e63538a2d435659d5361cd18e34005addcafb37",
            },
            Pin {
                architecture: "macos.h14c",
                tree: "d4aa54bec7ad98dc7a2fa884bb9f85b3ebb929459f25294b97a66ba2d2beb722",
                manifest: "d8b9f5367f546c416a511c0db78e3d49e522ae6285961fec4d89bd5412ac243d",
            },
            Pin {
                architecture: "macos.h14d",
                tree: "1b4c8d93f52f98c0884881249dec862b19c7982cba74c9e6a1e5e219ca1bcf83",
                manifest: "46738e306cbcd048e92bd404bc447456ce2fd684d664dce92b51109332cd1b1d",
            },
            Pin {
                architecture: "macos.h14g",
                tree: "8c1740fbf8ae61533ba92b501e1dd474daae101b4c21b4de29df36d46054229e",
                manifest: "8de0c74b19844a204371678278ac00fb939dd14e212baeda65cfd5ed8ec1a263",
            },
            Pin {
                architecture: "macos.h14s",
                tree: "b662cc13391596256ebbf95d47a1d5787282bbc1c66117f5bbe6830158182149",
                manifest: "73fb81dd786d08b82cf12d4e510714f37376154e2536999fe52e40925f842110",
            },
            Pin {
                architecture: "macos.h15c",
                tree: "b003b8cc39ec5042bd10663ad957f68bf768ff89c4c9098e457863aaf9f02bad",
                manifest: "64af945ea11c4a4bdae40e50fe62a5b7979b7ab562cde6034550cd15288cfd2e",
            },
            Pin {
                architecture: "macos.h15d",
                tree: "d2e057777360d02a1389b821a70210440ae4e6d2e7d574be24642a08e6fc76c7",
                manifest: "df5c1b1edd4c0bd95b0cc4f92eb4fb0db49736cad5865c2aa709bde396f43a9b",
            },
            Pin {
                architecture: "macos.h15g",
                tree: "a7b6a8e8a833f5ffc4b23be3e69c859e22d7dfe7efa4142951a59a6316be33ca",
                manifest: "42367cb474789726dce5a14dfd151330205434843075e3b4271f56c4899c69ae",
            },
            Pin {
                architecture: "macos.h15m",
                tree: "8de2d437b438020c8478ff8c8870948bf32231525a008f23963e07bf9e115299",
                manifest: "24d37adda2107b3d5cfc6d7b57a15312c291f1631facdd6aa6eace621c4ed3f9",
            },
            Pin {
                architecture: "macos.h15s",
                tree: "5685cb806281c576441e4008e8d40bbde2b0c7673748a96ae7d9d26c9a8d553b",
                manifest: "834580e3c1a59e17e3ed78782a97b18a52b1eb9f2f4b46f38a69f6f95a42d497",
            },
            Pin {
                architecture: "macos.h16c",
                tree: "7495beb57f4581e547ce97819c7269c5fd4692a0604f5adda619fbd9de6b1fd5",
                manifest: "26100261c5bb595dba933462163f5a83ed7bf91ddd9d1a936ebd42fcfb3e8489",
            },
            Pin {
                architecture: "macos.h16g",
                tree: "21f665b8f9c396ea65f6b6a085bfc9f4ea95c67410ab74251301859119f17f0d",
                manifest: "d82d81b19b7abbdb2777b061d6a5999823d6253b648c5b64104e1d181011e417",
            },
            Pin {
                architecture: "macos.h16s",
                tree: "3b42e8f83ad8a91769fa3b853c840298428e2589f09327a9bfd905099d17043d",
                manifest: "a0c6492e3c9bdf5d331ee5d26fd3cc19a26450af4902b8c0dfba744e8ccf73c3",
            },
            Pin {
                architecture: "macos.h17c",
                tree: "bb95aad00cde3a614ef65e68d6d5c064f85e9a51042befffe212720123599732",
                manifest: "0b1959e9fc01016554aa19ba068b081b1156d85e6368c5b7fe7d5aaada36e326",
            },
            Pin {
                architecture: "macos.h17d",
                tree: "7815aa5c14b446f860e96437547c735eebe895faaba604ac112707dbf06505f4",
                manifest: "3af071ae11bdde2f8fd3bcf4536fadc0fafcfa9e208676eb6b27c29cfd5dedd5",
            },
            Pin {
                architecture: "macos.h17g",
                tree: "d35a7b3a3477ae9a9503e6e99a43ad3b7de97474451f1bbf5ff505d3273c41ac",
                manifest: "d837dbf3aa74a4320064148aea40a909127c2a2928f5bcd229ec2f94b7f81e9a",
            },
            Pin {
                architecture: "macos.h17p",
                tree: "7e1729dfdb1c5160d7e6fc94cd14fa251adce305a81102b48ad9e829b506cc2e",
                manifest: "4ef8bfcc561608cd3d190d0b51e1ba60a7a94de491b2479d8a957e29bf6ca2ef",
            },
            Pin {
                architecture: "macos.h17s",
                tree: "79d751d08c9749c2b76e7a19cac5a61d65580aae73e0f4d4f0a2d69ec5b665d2",
                manifest: "fcd43436b44abeda462d4ef49246f4cc14ff7401539eacef75880cfd65a63594",
            },
            Pin {
                architecture: "macos.h18g",
                tree: "db10d6dd8defbd269547eba2318b77c9df517552ceb740df5f4f8983f35af498",
                manifest: "410a3e60d90c67c48180edd245902d9b9e3cf002995b2911ab8138c4d031f556",
            },
        ],
    },
    Bundle {
        model: ModelId::Mert330M,
        stem: "mert_v1_330m_fp16",
        pins: &[
            Pin {
                architecture: "macos.h13c",
                tree: "0bebd1a3818f602bde64f6dbb743883902c24205b681ef2e551adce3b7ca9ec6",
                manifest: "e857f2871699224e93ef6ebfc0ce48759c73d8014c197f4b680f52ef83f65d42",
            },
            Pin {
                architecture: "macos.h13d",
                tree: "832efc4a61f404189d1891bb69eed6f251c3aef98cbccc2a5d92c4f84f258019",
                manifest: "6010beb3a57a3265d54429715fb3107176c273c4b589301127a53de39ad268b9",
            },
            Pin {
                architecture: "macos.h13g",
                tree: "71f753f4c3dc1bb14624bb8d25feb3a9f69915c590200a36b4594f0186955d85",
                manifest: "39ac855626856327d5c5dbfdcf388260cc874dc84c2cd0dc0d0121c67837a1a7",
            },
            Pin {
                architecture: "macos.h13s",
                tree: "ff2c2093b48755720d23e286e2a1e66eeea53f4d3fed60548365d6c02c08d497",
                manifest: "0a0bb308d1b9272b560e12194beac39dbe4cae81f964bbeb0918a595abe7add1",
            },
            Pin {
                architecture: "macos.h14c",
                tree: "15d337da507a013bca113fc51c6a3be7350661673da444d0d7d07dcfa6a74a69",
                manifest: "75bdf04e3c540cf596f9e407f1db7892cfb06324e13afc4367c3024302369041",
            },
            Pin {
                architecture: "macos.h14d",
                tree: "0c9cfb1ff1b084998a889c7edba12791e520cce2c4a19ff5edec556a41759738",
                manifest: "0b1bdc59ca131cd4b4ead63eb5f91dc29c273ca84d74c479e31a9526a9702979",
            },
            Pin {
                architecture: "macos.h14g",
                tree: "35d87af914ca37147224863047dc90d3a7e5f80af319ea53e85cf4cf7039c446",
                manifest: "7e1770dc121c1eb1e7dce07275bba3e3a1e04772060f9f3e35215fe11dab635a",
            },
            Pin {
                architecture: "macos.h14s",
                tree: "f3f5267aa341319c37da783ed1786904863157b07b08873184abc121b3a4e7c5",
                manifest: "05c98c0efd0e0529cfed1bf245518ff5d1873b6827daf6fe93e7be52b6e0cfe1",
            },
            Pin {
                architecture: "macos.h15c",
                tree: "ae7687303beeea6301142ce823f37638c60c3207350c50f9e664fb91feb8ecc8",
                manifest: "8a01ff237fcc8fb192d2d0d9ebf8b299818aa260f7981f00e0d1f594cca13a23",
            },
            Pin {
                architecture: "macos.h15d",
                tree: "cd8e6b227e56c9fe00d93222b5b8aa886d84d27c11dfab55103878af31246e49",
                manifest: "3e1c3ab75242ff82b6c2b8ee24d515028cbc9f44547856da75dbd746bdb3397a",
            },
            Pin {
                architecture: "macos.h15g",
                tree: "441935d9e0719caa07132f1ccbda3a11d5414b1fd39a3311f6439cc7f094a1ea",
                manifest: "68751570a8b5b435f281caa8f91eb5271f7d2abd9b1307c539d8a80d0c2267d2",
            },
            Pin {
                architecture: "macos.h15m",
                tree: "01e3a3bdad86ab214fdaad7cb6a829ec989eb88cf2bb8910edf3a0b6f9929858",
                manifest: "cc95c31aca5397b965f2c5523039358728bb83fe9e9d8c152911b5ca3cf19fe1",
            },
            Pin {
                architecture: "macos.h15s",
                tree: "2cce213c23bbcc4c228beb0d391a45cf1a8bd6a9ee9ff5e545145f18e1ff5962",
                manifest: "dcddb7f97309c7b1337152ebe38680c47082208799ffc88120321536f4b7fb58",
            },
            Pin {
                architecture: "macos.h16c",
                tree: "f613968f77795a04e08f57b8df67c8d576f3e29fe72c000467960268b9ec1065",
                manifest: "c7b06b4b581fcc1295152ffc9e6ba2d31eac83cbda760a53fc3a9ff7fef7578c",
            },
            Pin {
                architecture: "macos.h16g",
                tree: "8833c98f66da132327cc00204e285bd0d6232a19ca3c75aee074a7dd72ba2df3",
                manifest: "6cbdda81eedcb651e72a94643ffbab060a96a82bac2425053c329c25b2f6243b",
            },
            Pin {
                architecture: "macos.h16s",
                tree: "086c63213293053cca0c591b072c87ea7ab6311fffa15d57b8ac74a166e7cbc0",
                manifest: "26f6fa3514ced4cbd802f0b77451881eb53b4007d9b2f9d8260be27ee203ecc1",
            },
            Pin {
                architecture: "macos.h17c",
                tree: "643112cc745cfd39b8692b31c5c6745ec23a104fe6670f2f1d0e4af05482ea42",
                manifest: "68727693ffa36b7c446033dd86cfa8795d34d8a1439ef00c2c312b6936394eb5",
            },
            Pin {
                architecture: "macos.h17d",
                tree: "b8a66e616abc267edf2f6722834b8c3cb463893c83584bce7847e4bb77bae93d",
                manifest: "5cf9a7ca7061f2042f45b0ece5791b63e4e725e3e65358c9b3bccae4e1e2250b",
            },
            Pin {
                architecture: "macos.h17g",
                tree: "52ac1cc0bbbab151e644be5c6c907793c80c931753bf5b6d3f3f35aca8680b55",
                manifest: "0aa53cca8536368fbde2cb8530750eacae61def7b0b0b06fa60f0a2d8011a84b",
            },
            Pin {
                architecture: "macos.h17p",
                tree: "eadb705a62f9c9b8bfe957b4ceff0eb9095f3239eb19a0736d97eb5a72b4a1ac",
                manifest: "eee416da7c4c1650d4f679a053a96972903a4c47099d74b1dc22960f87c3e5e5",
            },
            Pin {
                architecture: "macos.h17s",
                tree: "0efd0d694b0b6679812c301d293ae1a194dc164fbb1d3e523d4792a81f06672f",
                manifest: "af94ed25b8a80ea6a9dfc9773168dadcc85c3345759b7807fac9f6aad51572ed",
            },
            Pin {
                architecture: "macos.h18g",
                tree: "c793f937dfb76a0f152ca31c0d721f250c59cb3c303aa2040fd0ff0dadf3a1d0",
                manifest: "acd669767a104cc4fdaa87835d1b03935d940a89d7301391e05a1a7cf60bd315",
            },
        ],
    },
    Bundle {
        model: ModelId::BeatThis1,
        stem: "beat_this_1",
        pins: &[
            Pin {
                architecture: "macos.h13c",
                tree: "dc7b5bf1c55849434f35110dbe611a0807039ecc9f30cecff3c4e922ad2e3935",
                manifest: "b9ab389b10916eca16ef5c03c67025cd83b1acade7d8a350d0f624656528f741",
            },
            Pin {
                architecture: "macos.h13d",
                tree: "80d0366fd1da318b1a695665f90a3913efeb60d2f3b248f469c6a16493e57df5",
                manifest: "980b58a8da242129604c079eb3300788e8bcdf324b30b1c6b34e4cc6f64ed04c",
            },
            Pin {
                architecture: "macos.h13g",
                tree: "dc5d03e3620b423342a2eb6d161921ea7e45c7f5203558faa0227a74aab9a36c",
                manifest: "661eec24014afa9e5ec275c937073d7e23ae95efd0bc18def3db7a9663548aca",
            },
            Pin {
                architecture: "macos.h13s",
                tree: "f737df2175dd1701a43e54ef9a0ef0d7effe8d5e64fee1ef2511dd113aa21206",
                manifest: "07df44870f608700b901f89f5f57ee85742828a628b80c5adb17b7087334ead6",
            },
            Pin {
                architecture: "macos.h14c",
                tree: "f99570879a8e6ae3dde66e42c3e57507b3751637aef46634415114a82d823e7f",
                manifest: "4829fc57538ce43eb8aa4da84ff9d2f4a032a54390a89a16edf85c36051ccc62",
            },
            Pin {
                architecture: "macos.h14d",
                tree: "b3d4becc3af16b722a856f941dfcc7e3c3b43448f0e06aa85009277357cc215b",
                manifest: "a51b610a34098aadb2054881546cd5c08c3ef52be0f2b694a5fe503af383d84e",
            },
            Pin {
                architecture: "macos.h14g",
                tree: "c626a686776af1b225740ccd54590b7b465171342076dbfe44149df70ee9b9b9",
                manifest: "c7dd6073fac6ce326b57b99b09f5a2bcc1c07f6c430e13d3ef97bfa47a9c3fab",
            },
            Pin {
                architecture: "macos.h14s",
                tree: "6cc8464e47fcd74a66224161c737bb03c830a8b4516625641fdeba7d41743bf6",
                manifest: "8bd91f33a3f6d1828634276cfc1045947510fa33217e5a3d3b36652d458df1e4",
            },
            Pin {
                architecture: "macos.h15c",
                tree: "f107ba4cea20f128d2c5b8edbad3b4d4747e2b03f3fb1a7e0e86a1867ed99c8d",
                manifest: "0b33f73dac4d64f62c83c1eaea34f0863b08830c617a3f6177a98b58d6cd1347",
            },
            Pin {
                architecture: "macos.h15d",
                tree: "feafdd8e4f01fca27a95fcc5a3d0ace10a7aa95f015dce218d5fa1aa6e286971",
                manifest: "f3b373b162c9fbc00dd179dcc5a11e7be1f740f58026e825ead07746ed604f05",
            },
            Pin {
                architecture: "macos.h15g",
                tree: "2ef133de7f31bcfc1217e4c7a6a0b0ad60a00cdc6d6f6163d0430b0b9000d5a9",
                manifest: "cc542c6286b73ee5c13c9a9294b8f9619f303403fc9ba87de23c66485833c60a",
            },
            Pin {
                architecture: "macos.h15m",
                tree: "51c59062a8c77156ffd0c01bc6b3933c084819ee660c79728a3fadfee106863b",
                manifest: "569cd02bbf17850a28061771d7e8ca5c7fb80c3941c49f9112f2e335096a93ab",
            },
            Pin {
                architecture: "macos.h15s",
                tree: "bc1ba718b4d06309d9f3f950176722d92d5b70a031c3db5f92bc12389a83395e",
                manifest: "09a840ff2451714ef2322a0583c565b4faa26f23d4c1b40387376c0c6009dcc0",
            },
            Pin {
                architecture: "macos.h16c",
                tree: "b27e4e50623f4d1b875dd3de7dc9d815506ca67911a95bb6b31aff65d76bda0e",
                manifest: "a6ff37471e544697c052c4344e695961295f2cf9712bbd556d97d8b0bcb81f2b",
            },
            Pin {
                architecture: "macos.h16g",
                tree: "8dc4fec83205519dedf7062edde5d358ab1e4dc9b8c75e4734879131117a687d",
                manifest: "36dd0459df967a2568d31f7090bd55ce23365650773beb500fe73c78d95b7145",
            },
            Pin {
                architecture: "macos.h16s",
                tree: "537638873ccc84dfa115c13d7e11e7250c674f8ede7532b813fe9fcea0806e6a",
                manifest: "cc163f4ad487fcc90df73016ff6cd0b33ee7e318099668f16bac531728af374f",
            },
            Pin {
                architecture: "macos.h17c",
                tree: "624d9139ac090fa353d868b1d4a7fffd3c6f216610b3d01990d41eef87877ca7",
                manifest: "432ae97a9bc8deb7c15f7d602353e028cf0a7efbb81e41bf0c63a46f88ca6954",
            },
            Pin {
                architecture: "macos.h17d",
                tree: "558dbb701e199c2c9086385e8b8f30a8e9af9f7fedfb7a1deee92dee1bbf904e",
                manifest: "13c247d3d59b24242007819e0e7b95acf3e2c9914dc2ff3a3e0434f915aca32a",
            },
            Pin {
                architecture: "macos.h17g",
                tree: "88851e165ff2629ed303f3bde46f7af22275bd738e388dbe3f94e615dea0b7f7",
                manifest: "b59c3efcfbc8c9231e648dab767b16364bb8f95220faff62ce06af9f491b8870",
            },
            Pin {
                architecture: "macos.h17p",
                tree: "f49fb60cc10ab51d735d8f172ec6c42375ebb6ee28b87343dac4886a464a7790",
                manifest: "9a43475e38a99a87b05eb9d3667721c0867d3141fb4cc1843791cd5eb5d33bd1",
            },
            Pin {
                architecture: "macos.h17s",
                tree: "020789edab5f2c8782e4236cd1e1949dcfb9f408f44779fa7d8e110467ca6646",
                manifest: "87db7395c3a4bb62cff1c4f561aa9cc2cd0179d6be8047485a57d7ac8a65a687",
            },
            Pin {
                architecture: "macos.h18g",
                tree: "20c5034b735f8d7456897b8499efe06cf54b9e27392e886cb25c24e99beffc5e",
                manifest: "0b921aaecb155e5211c494e9581224e9c6641c6e66a0548cdc87627bf9b42b07",
            },
        ],
    },
    Bundle {
        model: ModelId::BeatThis2,
        stem: "beat_this_2",
        pins: &[
            Pin {
                architecture: "macos.h13c",
                tree: "8c3c0d384f7b3e8d02ee1efe650ceaf4779bc58bee9dfe6b0a0549b6c0e09b0a",
                manifest: "2378392433803e8dcb18b81a00d6764630dd5739e7aac9733fb54c53de43c98b",
            },
            Pin {
                architecture: "macos.h13d",
                tree: "370561c9f7152cfc49e5480db31e3ecb63783909191baf53d498d6be4be020ed",
                manifest: "5583a8c4a76505a1885ee538e4e1f46a2cf18176653bf738f154f4f943c05da0",
            },
            Pin {
                architecture: "macos.h13g",
                tree: "9fea5b5bd3ad2d6dfd4b4056147754bd4aceef006670c3d80c802ce70b0ff2b3",
                manifest: "a408069b90e71f4eba866ca484fd6940f6375cc340f764228552abcf60d5931b",
            },
            Pin {
                architecture: "macos.h13s",
                tree: "f047671093ebf4cafa4a40fcc2233c2e86ca466d0959dd0766e5d5c778432bd8",
                manifest: "ef759f5407de67b1018684ccbfd83567f71782efad0ea490c26adae6602d2ce5",
            },
            Pin {
                architecture: "macos.h14c",
                tree: "a44f31b8d4882a9c028e9b2b817efc53505b56a4b02f02ac8c1791a71bdcf387",
                manifest: "08c73b0412e17258d03596458c3916c57f7d785e269c87bf200ddd63c8017467",
            },
            Pin {
                architecture: "macos.h14d",
                tree: "f5e40c5fd4f0ab0b386713c7a382dde1fff33cf5112874c16e48934e26efcd7d",
                manifest: "61bdc2b9fe1c854e635e7ffe999aad44b3148fb693b3a5be6153023acb969200",
            },
            Pin {
                architecture: "macos.h14g",
                tree: "9fbdd21416933e0c3cd36fbd930cccf01754ef4b2029e4c38b96ca395aaa4322",
                manifest: "6fe2dfe52142b0ede334465f48bee40e1117826c4b1d27d5c4a8871467dc60ae",
            },
            Pin {
                architecture: "macos.h14s",
                tree: "69ed5cf0c5b664c453f933f76cc3b8a3b36fdda19762622d953cf3ed327caf5b",
                manifest: "f1a550c264243551a0e7caca6edf577a8efd617dc0baffb6ea82ecd92515fd80",
            },
            Pin {
                architecture: "macos.h15c",
                tree: "0e136c66819627e114605b8685cc0f2dd8249c79eaef1ee051b203c436b224e8",
                manifest: "9b702d54fa9d5f484899e6f22246849fb0f956019c1483971f4c84a17d2ded59",
            },
            Pin {
                architecture: "macos.h15d",
                tree: "5befa915c73c08ed827ab798eee1143d49bc4b2fdc66ac21e46d7e20370c6b0f",
                manifest: "47d71208891db2136545e04ac9be6a13753a82fd4bdd4dbf693cdde394df3de6",
            },
            Pin {
                architecture: "macos.h15g",
                tree: "67dcf95bff7e5bd3a34b6a0f305dec49665136c88c7ba8110b50ec6551b528c6",
                manifest: "c236f03df356412a9f29af8e58cb414985530eaf07ef58bd1a3832a19b24bd1a",
            },
            Pin {
                architecture: "macos.h15m",
                tree: "32ce74c4c87acb70fe50428ac9751c356e7664561267d42cfad84c3f7905d937",
                manifest: "c7d97230f09fa957cec3d63b0622b94ba16bfc55cf37d7cd22430bde51cdd1f6",
            },
            Pin {
                architecture: "macos.h15s",
                tree: "01d048553ea46d4686dc799025b310b1c16c6b8afa515f78c4949d436fb462fb",
                manifest: "5be8dacecb9eeb28cde640491a5390c0236b9a13c4161be4afc5c69628dae623",
            },
            Pin {
                architecture: "macos.h16c",
                tree: "bb337159754abb10861250b16718b381aa7d47dd1dc10f43831d9b640daa3b43",
                manifest: "1c603e3d08879a5e13ded35d210ebe7b812e1c3c97b25c2955c19a0e9a82690f",
            },
            Pin {
                architecture: "macos.h16g",
                tree: "62e7c684d816d2f5264fd014d5be2b16d26672a8201f91f96681a65ea1cf7e9a",
                manifest: "a794ce3561348b3b96ae468b937517fa9a58f3c23808d68628f063857d7a2d40",
            },
            Pin {
                architecture: "macos.h16s",
                tree: "bb1f5f187383711a25b8f632968d224e4db82c12fef589546e28220f8724a067",
                manifest: "1f7cec003b84a9bb2c45202e6a5ec6115f4d7189fadb8fc5795712b3dbeecca9",
            },
            Pin {
                architecture: "macos.h17c",
                tree: "56b44c6c202749c6a00f6a6b84785cecf459f7946d6d927a2e56060e778156d0",
                manifest: "055d3790a4e4ca1e7bc82901046ecc954d0fc4961374c54601a5c7616d021f8a",
            },
            Pin {
                architecture: "macos.h17d",
                tree: "ebf6afa3a13b96bf333a4b486f78cd9c3bb51e67d9e1dfc1b616b0a826e03184",
                manifest: "1594e22c69d3546c4dc1d62f40a6a8620e616dae50d6bbd5cf0a9114d98b6223",
            },
            Pin {
                architecture: "macos.h17g",
                tree: "fe9f6c1d322add8fa258f2a0f16ad4e644823a34b870e30f779d9f9fd30cbfe9",
                manifest: "fba73805b7650b648daef0b6a9ba03474ee4def584120492b6475c78c87f6bbd",
            },
            Pin {
                architecture: "macos.h17p",
                tree: "71fc3eddbb01b78d07434e8a401f9f9a7e32f7862596c0969847eeccfb0ca781",
                manifest: "04717bfb89591a3aea74a55d38ff8a807230d94baa6175ae749cc4359288675d",
            },
            Pin {
                architecture: "macos.h17s",
                tree: "4dc45da43f07cf8abaa8107b7b44dd5b34b1bd0057d7e2606d16ee0854bcd112",
                manifest: "668795734ea9c6ef150785758ae2eebea8a359d9d73ee2585ec14011272afb7c",
            },
            Pin {
                architecture: "macos.h18g",
                tree: "b0b9381dbf88134f623c1b15c379e94bd5c859608c17155d4dadc2dd2a0272e1",
                manifest: "4a941585815d80d96bfded755e310ac6711ac2a9a692ad395897df082fc7dff4",
            },
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use ferment_engine_test_fixtures::export;

    /// Kill: omit a chip/model, choose an iOS pin, or alter either digest.
    /// The expected objects come from the exporter's raw accepted contracts.
    #[test]
    fn every_macos_pin_matches_the_export_and_selects_only_one_chip() {
        let Some(_) = export::path("glap/model.json") else {
            return;
        };
        let families = [
            (
                "beat_this",
                &[
                    ("beat_this_1", ModelId::BeatThis1),
                    ("beat_this_2", ModelId::BeatThis2),
                ][..],
            ),
            (
                "glap",
                &[("audio", ModelId::GlapAudio), ("text", ModelId::GlapText)][..],
            ),
            (
                "siglip2",
                &[
                    ("image", ModelId::Siglip2Image),
                    ("text", ModelId::Siglip2Text),
                ][..],
            ),
            (
                "mert",
                &[
                    ("mert_v1_95m", ModelId::Mert95M),
                    ("mert_v1_330m", ModelId::Mert330M),
                ][..],
            ),
        ];
        let mut checked = 0;
        for (family, members) in families {
            let contract: serde_json::Value =
                serde_json::from_slice(&export::require_bytes(&format!("{family}/model.json")))
                    .unwrap();
            for &(member, id) in members {
                let source = &contract["members"][member]["artifacts"]["coreai"];
                let stem = source["file"]
                    .as_str()
                    .unwrap()
                    .strip_suffix(".aimodel")
                    .unwrap();
                let pins = source["pins"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|pin| pin["object"].as_str().unwrap().contains(".macos."))
                    .collect::<Vec<_>>();
                assert_eq!(pins.len(), 22, "{id}: qualified Mac chip inventory");
                for pin in pins {
                    let object = pin["object"].as_str().unwrap();
                    let arch = object
                        .strip_prefix(&format!("{stem}.macos."))
                        .unwrap()
                        .strip_suffix(".aimodelc")
                        .unwrap();
                    let manifest = crate::fleet_manifest_for_coreai(arch);
                    manifest.validate_complete_fleet().unwrap();
                    assert_eq!(manifest.placements(BackendKind::CoreAi).len(), 8);
                    let artifact = manifest
                        .model(id)
                        .unwrap()
                        .artifact_for(BackendKind::CoreAi)
                        .unwrap();
                    assert_eq!(artifact.file_name, object);
                    assert_eq!(artifact.sha256.as_deref(), pin["tree_sha256"].as_str());
                    assert_eq!(artifact.placement, Some(Placement::CoreAi));
                    let ArtifactSource::ManifestPrefix {
                        prefix_url,
                        manifest_sha256,
                    } = &artifact.source
                    else {
                        panic!("{id}: Core AI must use a verified manifest prefix");
                    };
                    assert_eq!(prefix_url, &format!("{MODEL_BASE_URL}/{object}"));
                    assert_eq!(
                        Some(manifest_sha256.as_str()),
                        pin["manifest_sha256"].as_str()
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 176);
    }

    /// Kill: invent a Core AI fallback artifact for an unsupported architecture.
    #[test]
    fn an_unknown_chip_retains_the_complete_base_fleet() {
        let selected = crate::fleet_manifest_for_coreai("unknown-chip");
        selected.validate_complete_fleet().unwrap();
        assert!(selected.placements(BackendKind::CoreAi).is_empty());
        assert_eq!(selected, crate::default_fleet_manifest());
    }
}
