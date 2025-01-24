## Usage

Please merge the files to get the complete model.

```bash
cat cc.zh.300.bin_part_* > cc.zh.300.bin
# The expected checksum is: 9219907cb26988af70d3dbf471ca777fb21082bd2ea70189f4ac4bed508d82cf
shasum -a 256 cc.zh.300.bin
```