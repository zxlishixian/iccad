# Catalog Hard-Gate Root Cause Map

Date: 2026-08-24

This file records the one-line root cause for each bug retained in the
853-case hard-gate candidate. The wording describes the injected RTL change,
not a label copied into any released log.

| Bug | Catalog class | One-line root cause |
|---|---|---|
| bug_128 | Branch | `bne` is decoded as `beq`. |
| bug_129 | Branch | Signed `blt` comparison uses the unsigned comparator. |
| bug_130 | Branch | Signed `bge` comparison uses the unsigned comparator. |
| bug_132 | Compressed branch | Compressed branch condition is inverted. |
| bug_133 | Jump/immediate | Negative `jal` immediate is zero-extended instead of sign-extended. |
| bug_134 | Jump/immediate | Negative `jalr` immediate is zero-extended instead of sign-extended. |
| bug_136 | Compressed jump | Compressed `jalr` link value increments by four instead of two. |
| bug_140 | Jump target | `jalr` target uses an S-type immediate instead of an I-type immediate. |
| bug_141 | Jump link | `jal` link calculation uses the register-base operand. |
| bug_142 | Jump link | `jalr` link calculation uses the register-base operand. |
| bug_143 | Compressed branch | Compressed branch source register selects the wrong bit. |
| bug_145 | MDU | `div` is decoded as `rem`. |
| bug_146 | MDU | `remu` uses signed operands. |
| bug_147 | LSU | Byte-load sign mode is inverted. |
| bug_148 | LSU | Halfword-load sign mode is inverted. |
| bug_149 | CSR/exception | `mepc` writes are over-aligned instead of preserving the architectural address. |
| bug_152 | MDU | `divu` uses signed operands. |
| bug_153 | LSU | Halfword store is encoded with byte-store type. |
| bug_160 | Register file | Read port A aliases register x31 to x30. |
| bug_161 | Register file | Read port B aliases register x30 to x29. |
| bug_162 | Writeback/store | Word store writes rs1 data instead of rs2 data. |
| bug_169 | LSU/writeback | Byte store at offset one selects the wrong data lane. |
| bug_170 | LSU/writeback | Halfword store at offset two selects the low lanes. |
| bug_172 | LSU/alignment | Late alignment/data-lane handling is corrupted for the targeted access pattern. |
| bug_173 | LSU/writeback | Byte load at offset three reads the wrong data lane. |
| bug_174 | LSU/writeback | Halfword load at offset two reads the low halfword. |
| bug_176 | LSU/writeback | Word-store offset-two rotation is incorrect. |
| bug_178 | Writeback | Writes to x31 are suppressed. |
| bug_179 | Writeback/register file | Writes to x30 alias x29. |
| bug_182 | RV32C/immediate | Negative `c.li` immediate is zero-extended. |
| bug_184 | RV32C/shift | `c.srli` is decoded as `c.srai`. |
| bug_185 | RV32C/shift | `c.srai` is decoded as `c.srli`. |
| bug_186 | RV32C/immediate | Negative `c.andi` immediate is zero-extended. |
| bug_193 | CSR/mstatus | Reading `mstatus.MPIE` returns the `MIE` value. |
| bug_194 | CSR/mstatus | Reading `mstatus.MPP` is forced to user mode. |
| bug_195 | CSR/mie | Software interrupt readback uses the timer bit. |
| bug_196 | CSR/mie | Timer interrupt readback uses the external bit. |
| bug_198 | CSR/mcause | The interrupt bit is cleared when reading `mcause`. |
| bug_202 | Debug CSR | Reading `dcsr` returns `dpc`. |
| bug_204 | Debug entry | Debug entry cause is forced to single-step. |
| bug_211 | Interrupt control | The exported `mstatus.MIE` value uses `MPIE`. |

## Usage Notes

- `bug_172` is intentionally described at the observable RTL-behavior level;
  its promoted cases are the repaired, collision-free alignment top-up.
- This map is documentation only. It does not alter `input.csv`, `golden.csv`,
  or any released log.
- Low-count bugs remain low-count when safe supplementary runs are clean or
  collide. Case count must not be increased by relaxing the hard gates.

