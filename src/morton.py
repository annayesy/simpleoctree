import numpy as np

### MORTON REFERENCE
# https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/

########################################## CONSTANTS #######################################

EIGHT_BIT_MASK = int('1' * 8,2)
NINE_BIT_MASK  = int('1' * 9,2)

LEVEL_OFFSET   = 52
LEVEL_MASK     = EIGHT_BIT_MASK << LEVEL_OFFSET

MAX_LEVEL2     = 25
X2_MASK        = int('10' * MAX_LEVEL2,2)
Y2_MASK        = int('01' * MAX_LEVEL2,2)

MAX_LEVEL3     = 16
X3_MASK        = int('100' * MAX_LEVEL3,2)
Y3_MASK        = int('010' * MAX_LEVEL3,2)
Z3_MASK        = int('001' * MAX_LEVEL3,2)

XY3_MASK       = X3_MASK | Y3_MASK
YZ3_MASK       = Y3_MASK | Z3_MASK
XZ3_MASK       = X3_MASK | Z3_MASK

# 2D encode and decode
mortonLUT_encode2d = np.zeros(256,int)
for j in range(1,256):
    bits  = list(np.binary_repr(j))
    nbits = len(bits)

    code  = ['0'] * (2*(nbits-1)+1)
    code[::2] = bits

    mortonLUT_encode2d[j] = int(''.join(code),2)

mortonLUT_decode2d    = np.zeros(256,int)
for j in range(1,256):
    bits   = list(np.binary_repr(j))
    if (np.mod(len(bits),2) == 0):
        bits = ['0'] + bits
    assert np.mod(len(bits),2) == 1
    decode = bits[::2]
    mortonLUT_decode2d[j] = int(''.join(decode),2)

# 3D encode and decode
mortonLUT_encode3d = np.zeros(512,int)
for j in range(1,512):
    bits  = list(np.binary_repr(j))
    nbits = len(bits)

    code  = ['0'] * (3*(nbits-1)+1)
    code[::3] = bits

    mortonLUT_encode3d[j] = int(''.join(code),2)

mortonLUT_decode3d    = np.zeros(512,int)
for j in range(1,512):
    bits   = list(np.binary_repr(j))
    if (np.mod(len(bits),3) == 0):
        bits = ['0'] + bits
    elif (np.mod(len(bits),3) == 2):
        bits = ['0','0'] + bits
    assert np.mod(len(bits),3) == 1
    decode = bits[::3]
    mortonLUT_decode3d[j] = int(''.join(decode),2)


########################################## END OF CONSTANTS #######################################

def get_root_params(XX):
    min_bound = np.min(XX,axis=0)
    max_bound = np.max(XX,axis=0)

    L0 = np.max(max_bound - min_bound)
    c0 = min_bound + 0.5 * L0
    return c0,L0+0.001

def get_key_params(key,c0,L0):
    lev     = get_level(key)

    box_len    = L0 / (1 << lev)
    box_anchor = keys_to_anchors(np.array([key],dtype=int),ndim=c0.shape[-1])[0]

    box_center = c0.copy() - 0.5*L0
    box_center += (box_anchor[:-1]+0.5) * box_len
    return box_center,box_len

def get_keys_centers(keys,c0,L0):
    lev     = get_level(keys)
    assert (np.linalg.norm( lev[0] - lev ) < 1e-15)
    lev     = lev[0]

    ndim = c0.shape[-1]

    box_len    = L0 / (1 << lev)
    box_anchors = keys_to_anchors(keys,ndim=ndim)

    box_centers  = (box_anchors[:,:-1]+0.5) * box_len
    box_centers += c0.copy() - 0.5*L0
    return box_centers

def encode_helper2d(x,encode_table):
    result = np.zeros(x.shape[0],dtype=int)

    result = result <<  0 | encode_table[(x >> 24) & EIGHT_BIT_MASK]
    result = result << 16 | encode_table[(x >> 16) & EIGHT_BIT_MASK]
    result = result << 16 | encode_table[(x >>  8) & EIGHT_BIT_MASK]
    result = result << 16 | encode_table[(x >>  0) & EIGHT_BIT_MASK]
    return result

def decode_helper2d(keys,decode_table):
    result = np.zeros(keys.shape[0],dtype=int)
    for j in range(7):
        result |= decode_table[keys >> (8*j) & EIGHT_BIT_MASK] << (4*j)
    return result

def encode_helper3d(x,encode_table):
    result = np.zeros(x.shape[0],dtype=int)

    result = result <<  0 | encode_table[(x >> 8) & EIGHT_BIT_MASK]
    result = result << 24 | encode_table[(x >> 0) & EIGHT_BIT_MASK]
    return result

def decode_helper3d(keys,decode_table):
    result = np.zeros(keys.shape[0],dtype=int)
    for j in range(7):
        result |= decode_table[keys >> (9*j) & NINE_BIT_MASK] << (3*j)
    return result

def get_level(keys):

    return keys >> LEVEL_OFFSET

def clear_level(keys):

    tmp = keys & ~(keys & LEVEL_MASK)
    return tmp

def set_level(keys,lev):

    return lev << LEVEL_OFFSET | keys

def points_to_anchors(points,level,c0,L0,ndim):
    if (ndim == 2):
        assert level < MAX_LEVEL2
    elif (ndim == 3):
        assert level < MAX_LEVEL3

    if (points.ndim == 1):
        points = np.expand_dims(points,axis=0)
    assert points.shape[-1] == ndim

    N = points.shape[0]
    anchor_root = c0 - L0 * 0.5

    len_level = L0 / (1 << level)

    anchors = (points-anchor_root) // len_level

    anchors = np.hstack((anchors,np.ones((N,1)) * level))
    return anchors.astype(int)

def anchors_to_keys(anchors,ndim):

    if (anchors.ndim == 1):
        anchors = np.expand_dims(anchors,axis=0)

    lev = anchors[:,-1]
    if (ndim == 2):

        x   = anchors[:,0]; y = anchors[:,1]
        keys_x = encode_helper2d(x,mortonLUT_encode2d)
        keys_y = encode_helper2d(y,mortonLUT_encode2d)

        keys = keys_x << 1 | keys_y
    else:
        x   = anchors[:,0]; y = anchors[:,1]; z = anchors[:,2]
        keys_x = encode_helper3d(x,mortonLUT_encode3d)
        keys_y = encode_helper3d(y,mortonLUT_encode3d)
        keys_z = encode_helper3d(z,mortonLUT_encode3d)

        keys = keys_x << 2 | keys_y << 1 | keys_z

    keys = set_level(keys,lev)
    return keys

def keys_to_anchors(keys,ndim):

    if (keys.ndim == 0):
        np.expand_dims(keys,axis=0)

    lev  = get_level(keys)
    keys = clear_level(keys)

    anchors = np.zeros((keys.shape[0],ndim+1),dtype=int)
    anchors[:,-1] = lev

    if (ndim == 2):

        anchors[:,0] = decode_helper2d(keys >> 1,mortonLUT_decode2d)
        anchors[:,1] = decode_helper2d(keys >> 0,mortonLUT_decode2d)

    else:
        anchors[:,0] = decode_helper3d(keys >> 2,mortonLUT_decode3d)
        anchors[:,1] = decode_helper3d(keys >> 1,mortonLUT_decode3d)
        anchors[:,2] = decode_helper3d(keys >> 0,mortonLUT_decode3d)
    return anchors

def get_parent(key,ndim):

    lev  = get_level(key)
    key  = clear_level(key)
    parent = key >> ndim

    parent = set_level(parent,lev-1)
    return parent

def get_ancestors(key,ndim):

    lev = get_level(key)
    ancestors = np.zeros(lev,dtype=int)

    for l in range(lev-1,-1,-1):
        key = get_parent(key,ndim=ndim)
        ancestors[l] = key
    return ancestors

def get_ancestors_vec(keys,ndim):

    lev = np.max(get_level(keys))
    ancestors = np.zeros((keys.shape[0],lev),dtype=int)

    for l in range(lev-1,-1,-1):
        keys = get_parent(keys,ndim=ndim)
        ancestors[:,l] = keys
    return ancestors

def get_siblings(key,ndim):

    lev  = get_level(key)
    if (lev == 0):
        return np.array([0],dtype=int)

    key = clear_level(key)

    nsibs    = 1 << ndim
    root     = (key >> ndim) << ndim

    siblings  = np.ones(nsibs,dtype=int) * root
    siblings  = siblings | np.arange(nsibs)
    siblings  = set_level(siblings,lev)
    return siblings

def get_siblings_vec(keys,ndim):

    lev  = get_level(keys)
    assert np.linalg.norm( (lev[0] - lev).astype(float) ) == 0
    lev = lev[0]; assert lev > 0

    keys = clear_level(keys)

    nsibs    = 1 << ndim
    roots    = (keys >> ndim) << ndim
    roots    = roots.reshape(roots.shape[0],1)

    siblings  = roots.repeat(nsibs,1)
    tmp       = np.arange(nsibs).reshape(nsibs,1).repeat(roots.shape[0],1).T
    siblings  = siblings | tmp
    siblings  = set_level(siblings,lev)
    return siblings

def get_children(key,ndim):
    lev = get_level(key)

    key = clear_level(key)

    key = set_level(key << ndim, lev+1)
    return get_siblings(key,ndim)

def increment_x(keys,ndim):

    lev  = get_level(keys)
    keys = clear_level(keys)

    if (ndim ==2):
        x_update = (keys | Y2_MASK) + 2
        result   = (x_update & X2_MASK) | (keys & Y2_MASK)
    else:
        x_update = (keys | YZ3_MASK) + 4
        result   = (x_update & X3_MASK) | (keys & YZ3_MASK)

    return set_level(result,lev)

def decrement_x(keys,ndim):

    lev  = get_level(keys)
    keys = clear_level(keys)

    if (ndim == 2):
        x_update = (keys & X2_MASK) - 2
        result   = (x_update & X2_MASK) | (keys & Y2_MASK)
    else:
        x_update = (keys & X3_MASK) - 4
        result   = (x_update & X3_MASK) | (keys & YZ3_MASK)
    return set_level(result,lev)

def increment_y(keys,ndim):

    lev  = get_level(keys)
    keys = clear_level(keys)

    if (ndim == 2):
        y_update = (keys | X2_MASK) + 1
        result   = (y_update & Y2_MASK) | (keys & X2_MASK)
    else:
        y_update = (keys | XZ3_MASK) + 2
        result   = (y_update & Y3_MASK) | (keys & XZ3_MASK)
    return set_level(result,lev)

def decrement_y(keys,ndim):

    lev  = get_level(keys)
    keys = clear_level(keys)

    if (ndim == 2):
        y_update = (keys & Y2_MASK) - 1
        result   = (y_update & Y2_MASK) | (keys & X2_MASK)
    else:
        y_update = (keys & Y3_MASK) - 2
        result   = (y_update & Y3_MASK) | (keys & XZ3_MASK)
    return set_level(result,lev)

def increment_z(keys,ndim):

    lev  = get_level(keys)
    keys = clear_level(keys)

    if (ndim == 2):
        assert ValueError("z only for 3d")
    else:
        z_update = (keys | XY3_MASK) + 1
        result   = (z_update & Z3_MASK) | (keys & XY3_MASK)
    return set_level(result,lev)

def decrement_z(keys,ndim):

    lev  = get_level(keys)
    keys = clear_level(keys)

    if (ndim == 2):
        assert ValueError("z only for 3d")
    else:
        z_update = (keys & Z3_MASK) - 1
        result   = (z_update & Z3_MASK) | (keys & XY3_MASK)
    return set_level(result,lev)

def get_colleagues_vec(keys,ndim):

    lev = get_level(keys)
    assert np.all(lev[0] == lev); lev = lev[0]
    if (lev == 0):
        return np.array([0],dtype=int)

    tmp    = keys.reshape(keys.shape[0],1)
    result = np.tile(tmp,(1,3**ndim) )

    if (ndim == 2):

        # decrement x
        inds = np.array([0,1,2],dtype=int)
        result[:,inds] = decrement_x(result[:,inds].copy(),ndim=2)

        # increment x
        inds = np.array([6,7,8],dtype=int)
        result[:,inds] = increment_x(result[:,inds].copy(),ndim=2)

        # decrement y
        inds = np.array([0,3,6],dtype=int)
        result[:,inds] = decrement_y(result[:,inds].copy(),ndim=2)

        # increment y
        inds = np.array([2,5,8],dtype=int)
        result[:,inds] = increment_y(result[:,inds].copy(),ndim=2)

        # check whether the computed anchors are valid and within box bounds
        max_val = (1 << lev) - 1
        anchor_bounds = np.array([[0,0,lev],\
                                 [max_val,0,lev],\
                                 [0,max_val,lev]])
        key_bounds = anchors_to_keys(anchor_bounds,ndim=2)

        tmp = result & (X2_MASK | LEVEL_MASK)
        valid_keys_x = np.logical_and(key_bounds[0] <= tmp, tmp <= key_bounds[1])
        tmp = result & (Y2_MASK | LEVEL_MASK)
        valid_keys_y = np.logical_and(key_bounds[0] <= tmp,tmp <= key_bounds[2])

        invalid_keys = np.logical_not( np.logical_and(valid_keys_x,valid_keys_y))
        result[invalid_keys] = -1
    else:
        # decrement x
        inds = np.arange(9)
        result[:,inds] = decrement_x(result[:,inds].copy(),ndim=3)

        # increment x
        inds += 18
        result[:,inds] = increment_x(result[:,inds].copy(),ndim=3)

        # decrement y
        inds = np.array([0,1,2,9,10,11,18,19,20],dtype=int)
        result[:,inds] = decrement_y(result[:,inds].copy(),ndim=3)

        # increment y
        inds += 6
        result[:,inds] = increment_y(result[:,inds].copy(),ndim=3)

        # decrement z
        inds = np.arange(0,27,3)
        result[:,inds] = decrement_z(result[:,inds].copy(),ndim=3)

        # increment z
        inds += 2
        result[:,inds] = increment_z(result[:,inds].copy(),ndim=3)

        # check whether the computed anchors are valid and within box bounds
        max_val = (1 << lev) - 1
        anchor_bounds = np.array([[0,0,0,lev],\
                                 [max_val,0,0,lev],\
                                 [0,max_val,0,lev],\
                                 [0,0,max_val,lev]])
        key_bounds = anchors_to_keys(anchor_bounds,ndim=3)

        tmp = result & (X3_MASK | LEVEL_MASK)
        valid_keys_x = np.logical_and(key_bounds[0] <= tmp, tmp <= key_bounds[1])
        tmp = result & (Y3_MASK | LEVEL_MASK)
        valid_keys_y = np.logical_and(key_bounds[0] <= tmp,tmp <= key_bounds[2])
        tmp = result & (Z3_MASK | LEVEL_MASK)
        valid_keys_z = np.logical_and(key_bounds[0] <= tmp,tmp <= key_bounds[3])

        invalid_keys = np.logical_not( np.logical_and(np.logical_and(valid_keys_x,valid_keys_y),\
                                                     valid_keys_z))
        result[invalid_keys] = -1
    return result

def get_colleagues(key,ndim):

    return get_colleagues_vec(np.array([key],dtype=int),ndim=ndim)[0]
