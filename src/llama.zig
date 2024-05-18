const z = @import("lib.zig");

const head_size = 128;
const num_heads = 32;
const num_kv_heads = 8;
const SOFTMAX_SCALE = @sqrt(@as(f32, head_size));
const H = num_heads * head_size;
const HKV = num_kv_heads * head_size;
const hidden_size = z.Dim{ .static = H };
const gqa_size = z.Dim{ .static = HKV };
const hidden_size2 = z.Dim{ .static = 2 * H };
const hidden_size3 = z.Dim{ .static = 3 * H };
const seqlen = z.Dim{ .dyn = 's' };
const device = z.Device.cpu;
const dtype = z.Dtype.f32;
const HiddenStates = Tensor(seqlen, hidden_size);
const Rotary = Tensor(seqlen, z.Dim{ .static = head_size / 2 });
const KState = Tensor(seqlen, gqa_size);
const CuSeqlen = z.Tensor(1, [1]z.Dim{seqlen}, device, z.Dtype.i32);
const HiddenStates2 = Tensor(seqlen, hidden_size2);
const HiddenStates3 = Tensor(seqlen, hidden_size3);

fn Tensor(comptime m: z.Dim, comptime n: z.Dim) type {
    return z.Tensor(2, [2]z.Dim{ m, n }, device, dtype);
}

const GateUpProj = Linear(hidden_size2, hidden_size);
const GateDown = Linear(hidden_size, hidden_size);
const Qkv = Linear(hidden_size3, hidden_size);
const Proj = Linear(hidden_size, hidden_size);

fn Linear(comptime m: z.Dim, comptime n: z.Dim) type {
    const Weight = z.Tensor(2, [2]z.Dim{ m, n }, device, dtype);
    return struct {
        weight: Weight,

        const Self = @This();
        fn zeros() Self {
            const weight = Weight.zeros();
            return Self{ .weight = weight };
        }

        fn forward(self: Self, comptime a: z.Dim, input: Tensor(a, n)) Tensor(a, m) {
            return input.matmul_t(m, self.weight);
        }
    };
}

const Mlp = struct {
    gate_up_proj: GateUpProj,
    gate_down: GateDown,

    const Self = @This();
    fn zeros() Self {
        const up_proj = GateUpProj.zeros();
        const down = GateDown.zeros();
        return Self{
            .gate_up_proj = up_proj,
            .gate_down = down,
        };
    }
    fn forward(self: Self, input: HiddenStates) HiddenStates {
        const gate_up: HiddenStates2 = self.gate_up_proj.forward(seqlen, input);
        const splits = gate_up.split2(.{ .dim = 1, .into = 2 });
        const gate: HiddenStates = splits[0];
        const up = splits[1];
        const tmp = gate.swiglu().mul(up);
        const out: HiddenStates = self.gate_down.forward(seqlen, tmp);
        return out;
    }
};

pub fn rotary_embed(q: HiddenStates, k: KState, cosin: Rotary) void {
    _ = cosin;
    _ = k;
    _ = q;
}

pub fn attention(q: HiddenStates, k: KState, v: KState, cu_seqlen: CuSeqlen, max_s: usize, softmax_scale: f32) t: {
    const Return = @TypeOf(q);
    break :t Return;
} {
    _ = softmax_scale;
    _ = max_s;
    _ = cu_seqlen;
    _ = v;
    _ = k;
    return q;
}

const Attention = struct {
    qkv: Qkv,
    o_proj: Proj,

    const Self = @This();
    fn zeros() Self {
        const qkv = Qkv.zeros();
        const o_proj = Proj.zeros();
        return Self{
            .qkv = qkv,
            .o_proj = o_proj,
        };
    }
    fn forward(self: Self, input: HiddenStates, cossin: Rotary, cu_seqlen: CuSeqlen, max_s: usize) HiddenStates {
        const qkv: HiddenStates3 = self.qkv.forward(seqlen, input);
        const splits = qkv.split(1, 3, [3]z.Dim{ hidden_size, gqa_size, gqa_size });
        const q: HiddenStates = splits[0];
        const k: KState = splits[1];
        const v: KState = splits[2];
        rotary_embed(q, k, cossin);
        // z.paged_attention(k, v, k_cache, v_cache, slots);
        const attn_output = attention(q, k, v, cu_seqlen, max_s, SOFTMAX_SCALE);
        const out: HiddenStates = self.o_proj.forward(seqlen, attn_output);
        return out;
    }
};

pub fn main() !void {
    const hidden_states = HiddenStates.zeros();
    const mlp = Mlp.zeros();
    const attn = Attention.zeros();
    const cossin = undefined;
    const cu_seqlen = undefined;
    const max_s = 1000;
    const final: HiddenStates = attn.forward(hidden_states, cossin, cu_seqlen, max_s);
    const final2: HiddenStates = mlp.forward(final);
    _ = final2;
}
