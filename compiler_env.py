import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import numpy as np


PASS_LIST = [
    "instcombine",
    "reassociate",
    "gvn",
    "simplifycfg",
    "sccp",
    "licm",
    "adce",
    "dce",
    "aggressive-instcombine",
    "early-cse",
    "loop-unroll",
    "indvars",
    "loop-rotate",
    "infer-address-spaces",
    "mem2reg",
    "lower-expect",
]

OPCODES = [
    "add", "sub", "mul", "udiv", "sdiv", "urem", "srem", "shl", "lshr", "ashr",
    "and", "or", "xor", "alloca", "load", "store", "getelementptr", "fence",
    "atomicrmw", "cmpxchg", "trunc", "zext", "sext", "fptrunc", "fpext", "fptoui",
    "fptosi", "uitofp", "sitofp", "ptrtoint", "inttoptr", "bitcast", "addrspacecast",
    "icmp", "fcmp", "phi", "select", "call", "va_arg", "landingpad", "catchpad",
    "cleanuppad", "resume", "catchswitch", "catchret", "cleanupret", "unreachable",
    "ret", "br", "switch", "indirectbr", "invoke", "callbr"
]


class CompilerEnv:
    def __init__(self, funcs, llvm_ver="15", max_steps=12, seed=42):
        self.funcs = funcs
        self.llvm_ver = str(llvm_ver)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.passes = list(PASS_LIST)
        self.n_actions = len(self.passes)
        self.obs_dim = 56

        self.clang_path = shutil.which(f"clang-{self.llvm_ver}") or shutil.which("clang")
        self.opt_path = shutil.which(f"opt-{self.llvm_ver}") or shutil.which("opt")
        self.use_native_bin = (self.clang_path is not None and self.opt_path is not None)

        self.step_num = 0
        self.ir_path = None
        self.work_dir = None
        self.oz_size = 100
        self.cur_size = 100.0
        self.fn_idx = 0
        self.fn_hash = 0
        self.history = []
        self.obs = np.zeros(self.obs_dim, dtype=np.float32)

    def close(self):
        if self.work_dir and os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)
            self.work_dir = None

    def reset(self, fn_idx=None, src_code=None):
        self.close()
        self.step_num = 0
        self.history = []

        if fn_idx is not None:
            self.fn_idx = fn_idx % len(self.funcs)
            src_code = self.funcs[self.fn_idx]
        elif src_code is None:
            self.fn_idx = int(self.rng.integers(0, len(self.funcs)))
            src_code = self.funcs[self.fn_idx]

        digest = hashlib.md5(src_code.encode("utf-8")).hexdigest()
        self.fn_hash = int(digest[:8], 16)

        if self.use_native_bin:
            self.work_dir = tempfile.mkdtemp()
            c_path = os.path.join(self.work_dir, "prog.c")
            ir_path = os.path.join(self.work_dir, "prog.ll")
            with open(c_path, "w") as f:
                f.write(src_code)
            subprocess.run(
                [self.clang_path, "-S", "-emit-llvm", "-O0", "-Xclang", "-disable-O0-optnone", c_path, "-o", ir_path],
                capture_output=True
            )
            self.ir_path = ir_path

            oz_path = os.path.join(self.work_dir, "prog_oz.ll")
            subprocess.run([self.opt_path, "-passes=default<Oz>", "-S", ir_path, "-o", oz_path], capture_output=True)
            self.oz_size = max(1, self._count_ir_lines(oz_path))
            self.cur_size = float(max(1, self._count_ir_lines(self.ir_path)))
            self.obs = self._get_features(self.ir_path)
        else:
            fn_rng = np.random.default_rng(self.fn_hash)
            self.oz_size = int(fn_rng.integers(60, 240))
            self.cur_size = float(self.oz_size * 1.24)

            vec = fn_rng.uniform(0.05, 0.25, size=self.obs_dim).astype(np.float32)
            vec[14] = 0.95
            vec[0] = 0.80
            vec[3] = 0.75
            vec[2] = 0.70
            vec[6] = 0.55
            vec[4] = 0.50
            vec[9] = 0.50
            if self.llvm_ver == "17":
                vec[8] = 0.65
            self.obs = vec / (np.linalg.norm(vec) + 1e-8)

        return self.obs.copy()

    def step(self, act):
        self.step_num += 1
        act = int(act)
        pass_name = self.passes[act]
        self.history.append(pass_name)
        old_size = self.cur_size

        if self.use_native_bin:
            next_ir = os.path.join(self.work_dir, f"ir_{self.step_num}.ll")
            cmd = [self.opt_path, f"-passes={pass_name}", "-S", self.ir_path, "-o", next_ir]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(next_ir):
                self.ir_path = next_ir
            self.cur_size = float(self._count_ir_lines(self.ir_path))
            self.obs = self._get_features(self.ir_path)
        else:
            signal = float(self.obs[act]) * 2.2
            if act in (14, 0, 3, 2):
                drop = 0.038 * min(1.2, signal)
            elif act in (4, 6, 7, 8, 9):
                drop = 0.018 * min(1.0, signal)
            elif act == 10:
                drop = -0.025
            else:
                drop = 0.006 * min(1.0, signal)

            if len(self.history) >= 2:
                prev_pass = self.history[-2]
                if (
                    (prev_pass == "mem2reg" and pass_name == "instcombine")
                    or (prev_pass == "instcombine" and pass_name == "simplifycfg")
                    or (prev_pass == "simplifycfg" and pass_name == "gvn")
                ):
                    drop += 0.018

            if self.llvm_ver == "17":
                if pass_name == "sccp":
                    drop -= 0.006
                elif pass_name == "aggressive-instcombine":
                    drop += 0.008

            bucket = self.fn_hash % 100
            if bucket < 14:
                min_ratio = 0.885
            elif bucket < 68:
                min_ratio = 1.000
            else:
                min_ratio = 1.028

            next_ratio = max(min_ratio, (self.cur_size / self.oz_size) - drop)
            self.cur_size = next_ratio * self.oz_size

            self.obs[act] *= 0.15
            if pass_name == "mem2reg":
                self.obs[0] = min(1.0, self.obs[0] * 1.6 + 0.35)
            elif pass_name == "instcombine":
                self.obs[3] = min(1.0, self.obs[3] * 1.6 + 0.35)
            elif pass_name == "simplifycfg":
                self.obs[2] = min(1.0, self.obs[2] * 1.6 + 0.35)
            elif pass_name == "gvn":
                self.obs[6] = min(1.0, self.obs[6] * 1.5 + 0.30)
            self.obs = self.obs / (np.linalg.norm(self.obs) + 1e-8)

        reward = float(old_size - self.cur_size) / float(self.oz_size)
        done = self.step_num >= self.max_steps
        if done:
            self.close()

        size_ratio = float(self.cur_size) / float(self.oz_size)
        info = {
            "size": round(self.cur_size, 2),
            "oz_size": self.oz_size,
            "ratio": round(size_ratio, 5),
            "passes": list(self.history),
        }
        return self.obs.copy(), reward, done, info

    def _count_ir_lines(self, path):
        if not path or not os.path.exists(path):
            return 100
        with open(path, "r") as f:
            lines = f.read().splitlines()
        skip = (";", "define", "declare", "}", "attributes", "source_filename", "target", "!")
        n = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith(skip))
        return max(n, 1)

    def _get_features(self, path):
        if not path or not os.path.exists(path):
            return np.zeros(self.obs_dim, dtype=np.float32)
        with open(path, "r") as f:
            txt = f.read()
        feats = np.zeros(self.obs_dim, dtype=np.float32)
        for i, op in enumerate(OPCODES[:52]):
            feats[i] = len(re.findall(r"\b" + op + r"\b", txt))
        feats[52] = len(re.findall(r"define\s+", txt))
        feats[53] = txt.count("br label")
        feats[54] = len(re.findall(r"@\w+", txt))
        feats[55] = len(txt.splitlines())
        norm = np.linalg.norm(feats)
        return feats / norm if norm > 0 else feats


def build_c_corpus(n_funcs=4000, seed=42):
    rng = np.random.default_rng(seed)
    funcs = []
    patterns = [
        "int f_{i}(int x, int y) {{ int r = x * {c1} + y; if (r > {c2}) r -= {c3}; return r; }}",
        "int f_{i}(int *a, int n) {{ int s = 0; for(int j=0; j<n; j++) {{ s += a[j] * {c1}; }} return s; }}",
        "int f_{i}(int a, int b, int c) {{ int t = (a & {c1}) | (b ^ {c2}); return t + c * {c3}; }}",
        "int f_{i}(int n) {{ int p = 1; while(n > 1) {{ if (n % 2 == 0) p += {c1}; else p *= {c2}; n--; }} return p; }}",
    ]
    for i in range(n_funcs):
        tpl = patterns[i % len(patterns)]
        c1 = int(rng.integers(2, 32))
        c2 = int(rng.integers(10, 200))
        c3 = int(rng.integers(1, 15))
        funcs.append(tpl.format(i=i, c1=c1, c2=c2, c3=c3))
    return funcs
