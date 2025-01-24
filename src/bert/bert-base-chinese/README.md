## Usage

Please merge the files to get the complete model.

```bash
cat pytorch_model.bin_part_* > pytorch_model.bin
# The expected checksum is: 8a693db616eaf647ed2bfe531e1fa446637358fc108a8bf04e8d4db17e837ee9
shasum -a 256 pytorch_model.bin
```