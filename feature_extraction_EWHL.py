import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix, diags, eye, issparse
import warnings
warnings.filterwarnings('ignore')


def build_edh_simplicial_complex(fc_matrix, threshold=0.5, epsilon=1e-8):
    n_nodes = fc_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    edges = []
    edge_weights = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            w = fc_matrix[i, j]
            if w >= threshold:
                edges.append((i, j))
                edge_weights.append(max(float(w), 1e-6))
                G.add_edge(i, j, weight=w)

    n_edges = len(edges)
    edge_to_idx = {(u, v): i for i, (u, v) in enumerate(edges)}

    triangles = []
    valid_tri_weights = []
    seen = set()

    for (u, v) in edges:
        try:
            neighbors = list(nx.common_neighbors(G, u, v))
        except:
            continue

        for w_node in neighbors:
            if w_node <= v:
                continue
            tri = tuple(sorted((u, v, w_node)))
            if tri in seen:
                continue
            seen.add(tri)

        
            
            triangles.append(tri)
            w1 = max(fc_matrix[u, v], 1e-6)
            w2 = max(fc_matrix[v, w_node], 1e-6)
            w3 = max(fc_matrix[u, w_node], 1e-6)
            valid_tri_weights.append((w1, w2, w3))

    n_triangles = len(triangles)

    rows_d0, cols_d0, data_d0 = [], [], []
    for i, (u, v) in enumerate(edges):
        rows_d0.extend([i, i])
        cols_d0.extend([u, v])
        data_d0.extend([-1, 1])
    d0 = csr_matrix((data_d0, (rows_d0, cols_d0)), shape=(n_edges, n_nodes)) if n_edges > 0 else csr_matrix((0, n_nodes))

    rows_d1, cols_d1, data_d1 = [], [], []
    for i, (a, b, c) in enumerate(triangles):
        try:
            idx1 = edge_to_idx[(a, b)]
            idx2 = edge_to_idx[(b, c)]
            idx3 = edge_to_idx[(a, c)]
            rows_d1.extend([i, i, i])
            cols_d1.extend([idx1, idx2, idx3])
            data_d1.extend([1, 1, -1])
        except:
            continue
    d1 = csr_matrix((data_d1, (rows_d1, cols_d1)), shape=(n_triangles, n_edges)) if n_triangles > 0 else csr_matrix((0, n_edges))

    star0 = eye(n_nodes)
    star0_inv = eye(n_nodes)
    star1 = diags(edge_weights) if n_edges > 0 else csr_matrix((0, 0))
    star1_inv = diags(1.0 / (np.array(edge_weights) + epsilon)) if n_edges > 0 else csr_matrix((0, 0))

    star2 = csr_matrix((0, 0))
    star2_inv = csr_matrix((0, 0))
    if n_triangles > 0:
        tri_weights = []
        for w1, w2, w3 in valid_tri_weights:
            tweight = (w1 * w2 * w3) ** (1/3)
            tri_weights.append(max(tweight, 1e-6))
        star2 = diags(tri_weights)
        star2_inv = diags(1.0 / (np.array(tri_weights) + epsilon))

    return {
        "n_nodes": n_nodes, "n_edges": n_edges, "n_triangles": n_triangles,
        "d0": d0, "d1": d1,
        "star0": star0, "star1": star1, "star2": star2,
        "star0_inv": star0_inv, "star1_inv": star1_inv, "star2_inv": star2_inv
    }


def verify_exterior_derivative(sc):
    d0, d1 = sc["d0"], sc["d1"]
    n_edges = sc["n_edges"]
    if n_edges == 0 or d1.shape[0] == 0:
        return True, 0.0
    d1d0 = d1 @ d0
    max_err = np.abs(d1d0).max()
    return max_err < 1e-10, max_err


def compute_edh_laplacians_and_energies(sc):
    d0, d1 = sc["d0"], sc["d1"]
    star0, star1, star2 = sc["star0"], sc["star1"], sc["star2"]
    star0_inv, star1_inv, star2_inv = sc["star0_inv"], sc["star1_inv"], sc["star2_inv"]
    n_nodes, n_edges = sc["n_nodes"], sc["n_edges"]

    if n_edges == 0:
        emp0 = csr_matrix((n_nodes, n_nodes))
        emp1 = csr_matrix((0, 0))
        return emp0, emp1, emp0, emp1, 0.0

    try:
        d0_star = star0 @ d0.T @ star1
        d1_star = star1_inv @ d1.T @ star2 if sc["n_triangles"] > 0 else csr_matrix((n_edges, n_edges))
        L0 = d0_star @ d0
        term1 = d0 @ d0_star
        term2 = d1_star @ d1 if sc["n_triangles"] > 0 else csr_matrix((n_edges, n_edges))
        L1 = term1 + term2
        sym_L0 = star0 @ L0
        sym_L1 = star1 @ L1
        tr_L1 = np.sum(L1.diagonal())
        tr_term2 = np.sum(term2.diagonal())
        curl = tr_term2 / (tr_L1 + 1e-10) if tr_L1 > 1e-10 else 0.0
    except:
        L0 = csr_matrix((n_nodes, n_nodes))
        L1 = csr_matrix((n_edges, n_edges))
        sym_L0 = L0
        sym_L1 = L1
        curl = 0.0
    return L0, L1, sym_L0, sym_L1, curl


def extract_enhanced_spectral_features(sym_L, zero_tol=1e-3):
    try:
        if sym_L.shape[0] == 0:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        mat = sym_L.toarray() if issparse(sym_L) else sym_L
        eigvals = np.linalg.eigvalsh(mat)
        eigvals = np.sort(eigvals)
        zero_mask = np.abs(eigvals) < zero_tol
        beta = int(np.sum(zero_mask))
        non_zero = eigvals[~zero_mask]
        if len(non_zero) == 0:
            return beta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        fiedler = non_zero[0]
        mean = np.mean(non_zero)
        std = np.std(non_zero)
        sumv = np.sum(non_zero)
        sqsum = np.sum(non_zero**2)
        rang = non_zero[-1] - non_zero[0]
        cnt = len(non_zero)
        return beta, float(fiedler), float(mean), float(std), float(sumv), float(sqsum), float(rang), int(cnt)
    except:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0


def extract_edh_features_single_subject(fc_matrix, file_id, lap_dir, thresholds):
    all_feats = []
#    curls = []
    log = ""
    saved = False

    for i, t in enumerate(thresholds):
        try:
            sc = build_edh_simplicial_complex(fc_matrix, t)
            if i == 0:
                ok, err = verify_exterior_derivative(sc)
                log = f"ok={ok}, err={err:.2e}"
 #               L0, L1, _, _, _ = compute_edh_laplacians_and_energies(sc)
#                save_laplacian_to_txt(L0, os.path.join(lap_dir, f"{file_id}_L0.txt"))
 #               save_laplacian_to_txt(L1, os.path.join(lap_dir, f"{file_id}_L1.txt"))
#                saved = True
            L0, L1, sL0, sL1, cr = compute_edh_laplacians_and_energies(sc)
            b0, f0, m0, s0, su0, sq0, r0, c0 = extract_enhanced_spectral_features(sL0)
            b1, f1, m1, s1, su1, sq1, r1, c1 = extract_enhanced_spectral_features(sL1)
            all_feats.extend([b0,f0,sq0,r0,c0, b1,f1,sq1,r1,c1,cr])
#            curls.append(cr)
        except:
            all_feats.extend([0]*11)
#            curls.append(0.0)

    try:
        final = np.array(all_feats)
    except:
        final = np.zeros(len(thresholds)*11)
    return final, saved, log


def load_fc_matrix(file_id, matrix_dir):
    names = [
        f"{file_id}_rois_aal.1D.txt",
        f"{file_id}_rois_aal.txt",
        f"{file_id}.1D.txt",
        f"{file_id}.txt"
    ]
    path = None
    for n in names:
        p = os.path.join(matrix_dir, n)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"FC文件不存在: {file_id}")
    d = np.loadtxt(path)
    m = d.reshape(116, 116)
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    m = np.clip(m, 1e-6, None)
    return m


def generate_enhanced_feature_names(thresholds):
    names = []
    for t in thresholds:
        ts = f"{t:.2f}".rstrip("0").rstrip(".")
        names += [f"t{ts}_beta0",f"t{ts}_F0",f"t{ts}_nz_sq_sum0",f"t{ts}_nz_range0",f"t{ts}_nz_count0"]
        names += [f"t{ts}_beta1",f"t{ts}_F1",f"t{ts}_nz_sq_sum1",f"t{ts}_nz_range1",f"t{ts}_nz_count1"]
        names += [f"t{ts}_curl_ratio"]
    # for p in ["mean","std"]:
    #     for d in ["0","1"]:
    #         names += [f"{p}_beta{d}",f"{p}_F{d}",f"{p}_nz_mean{d}",f"{p}_nz_std{d}",f"{p}_nz_sum{d}",f"{p}_nz_sq_sum{d}",f"{p}_nz_range{d}",f"{p}_nz_count{d}"]
    # names += ["mean_curl_ratio","std_curl_ratio"]
    return names


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LABEL_FILE = os.path.join(BASE_DIR, "884data.xlsx")
    MATRIX_DIR = os.path.join(BASE_DIR, "rois_aal_coor_matrix")
    OUTPUT_DIR = BASE_DIR
    LAP_DIR = os.path.join(BASE_DIR, "laplacians")

    THRESHOLDS = np.arange(0.3, 0.91, 0.05)
    os.makedirs(LAP_DIR, exist_ok=True)

    f_names = generate_enhanced_feature_names(THRESHOLDS)
    out_xlsx = os.path.join(OUTPUT_DIR, "884_subjects_143features.xlsx")
    header = ["Subject_ID", "Label"] + f_names

    df = pd.read_excel(LABEL_FILE)
    ids = df["FILE_ID"].astype(str).tolist()
    labels = df["DX_GROUP"].tolist()

    ids = ids[:884]
    labels = labels[:884]
   

    rows = []

    success = 0
    total = len(ids)


    for idx, (sid, lab) in enumerate(zip(ids, labels)):
        print(f"[{idx+1}/{total}] {sid:<25} ", end="")
        try:
            fc = load_fc_matrix(sid, MATRIX_DIR)
            feat, saved, log = extract_edh_features_single_subject(fc, sid, LAP_DIR, THRESHOLDS)
            row = [sid, lab] + feat.tolist()
            rows.append(row)
            success += 1
            print(f"✅ 成功 | {log}")
        except Exception as e:
            print(f"❌ 失败 | {str(e)[:60]}")

    pd.DataFrame(rows, columns=header).to_excel(out_xlsx, index=False)

    print("\n" + "="*50)
    print(f"测试完成！成功：{success}/{total}")
    print(f"输出Excel：{out_xlsx}")
    print("="*50)

if __name__ == "__main__":
    main()
