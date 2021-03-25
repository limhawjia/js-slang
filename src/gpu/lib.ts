import { GPU } from 'gpu.js'
import { TypeError } from '../utils/rttc'
import { parse } from 'acorn'
import { generate } from 'astring'
import * as es from 'estree'
import { gpuRuntimeTranspile } from './transfomer'
import { ACORN_PARSE_OPTIONS } from '../constants'

// Heuristic : Only use GPU if array is bigger than this
const MAX_SIZE = 200

// helper function to build 2D array output, modifies and returns res
function buildArray(arr: any, ctr: any, end: any, mem: any, ext: any, res: any) {
  const endMap = {}
  for (let i = 0; i < ctr.length; i++) {
    endMap[ctr[i]] = end[i]
  }

  if (mem.length == 0) {
    return arr
  }

  const cur = mem[0]
  if (typeof cur === 'string' && ctr.includes(cur)) {
    for (let i = 0; i < endMap[cur]; i++) {
      res[i] = buildArray(arr[i], ctr, end, mem, ext, res[i])
    }
  } else if (typeof cur === 'string' && cur in ext) {
    const v = ext[cur]
    res[v] = buildArray(arr[v], ctr, end, mem, ext, res[v])
  } else if (typeof cur === 'number') {
    res[cur] = buildArray(arr, ctr, end, mem, ext, res[cur])
  } else {
    // TODO: handle this properly
    throw 'Index is not number, counter or external variable'
  }

  return res
}

/*
 * we only use the gpu if:
 * 1. we are working with numbers
 * 2. we have a large array (> 100 elements)
 */
function checkValidGPU(f: any, end: any): boolean {
  let res: any
  if (end.length === 1) res = f(0)
  if (end.length === 2) res = f(0, 0)
  if (end.length === 3) res = f(0, 0, 0)

  // we do not allow array assignment
  // we expect the programmer break it down for us
  if (typeof res !== 'number') {
    return false
  }

  let cnt = 1
  for (const i of end) {
    cnt = cnt * i
  }

  return cnt > MAX_SIZE
}

// just run on js!
function manualRun(f: any, end: any, res: any) {
  function build() {
    for (let i = 0; i < end[0]; i++) {
      res[i] = f(i)
    }
    return
  }

  function build2D() {
    for (let i = 0; i < end[0]; i = i + 1) {
      for (let j = 0; j < end[1]; j = j + 1) {
        res[i][j] = f(i, j)
      }
    }
    return
  }

  function build3D() {
    for (let i = 0; i < end[0]; i = i + 1) {
      for (let j = 0; j < end[1]; j = j + 1) {
        for (let k = 0; k < end[2]; k = k + 1) {
          res[i][j][k] = f(i, j, k)
        }
      }
    }
    return
  }

  if (end.length === 1) return build()
  if (end.length === 2) return build2D()
  return build3D()
}

// helper function to build id map for runtime transpile
function buildRuntimeMap(ctr: any, mem: any) {
  const ids = []
  for (let m of mem) {
    if (typeof m === 'string' && ctr.includes(m)) {
      ids.push(m)
    }
  }

  if (ids.length > 3) {
    // TODO: handle this properly
    throw 'Identifiers in array indices should not exceed 3'
  }

  const t = [
    ['this.thread.x'],
    ['this.thread.y', 'this.thread.x'],
    ['this.thread.z', 'this.thread.y', 'this.thread.x']
  ]
  const threads = t[ids.length]

  const idMap = {}
  for (let i = 0; i < ids.length; i++) {
    idMap[ids[i]] = threads[i]
  }

  return idMap
}

// helper function to calculate setOutput array
function getRuntimeDim(ctr: any, end: any, mem: any) {
  const endMap = {}
  for (let i = 0; i < ctr.length; i++) {
    endMap[ctr[i]] = end[i]
  }

  const dimensions = []
  for (let m of mem) {
    if (typeof m === 'string' && m in endMap) {
      dimensions.push(endMap[m])
    }
  }

  return dimensions
}

// helper function to check dimensions of array
function checkArr(arr: any, ctr: any, end: any, mem: any, ext: any) {
  const endMap = {}
  for (let i = 0; i < ctr.length; i++) {
    endMap[ctr[i]] = end[i]
  }

  let ok = true
  let arrQ = [arr]

  for (let m of mem) {
    if (typeof m === 'number') {
      // current level of arrays need to be of at least length m + 1
      const newArrQ = []
      for (let a of arrQ) {
        if (!Array.isArray(a) || a.length <= m) {
          ok = false
          break
        }
        newArrQ.push(a[m])
      }
      arrQ = newArrQ
    } else if (ctr.includes(m)) {
      // current level of arrays need to be at least length endMap[m] + 1
      const newArrQ = []
      for (let a of arrQ) {
        if (!Array.isArray(a) || a.length <= endMap[m]) {
          ok = false
          break
        }
        for (let i = 0; i <= endMap[m]; i++) {
          newArrQ.push(a[i])
        }
      }
      arrQ = newArrQ
    } else if (m in ext) {
      const v = ext[m]
      if (typeof v !== 'number') {
        // TODO: handle this properyl
        throw 'External variable used as index is not a number'
      }
      const newArrQ = []
      for (let a of arrQ) {
        if (!Array.isArray(a) || a.length <= v) {
          ok = false
          break
        }
        newArrQ.push(a[v])
      }
      arrQ = newArrQ
    } else {
      // TODO: handle this properly
      throw 'Index should not be a local variable'
    }
    if (!ok) {
      break
    }
  }

  return ok
}

/* main function that runs code on the GPU (using gpu.js library)
 * @ctr: names of counters
 * @end : end bounds for counters
 * @mem: names/value of indices used in array assiginment
 * @extern : external variable definitions {}
 * @f : function run as on GPU threads
 * @arr : array to be written to
 */
export function __createKernel(
  ctr: any,
  end: any,
  mem: any,
  extern: any,
  f: any,
  arr: any,
  f2: any
) {
  const gpu = new GPU()

  if (!checkArr(arr, ctr, end, mem, extern)) {
    throw new TypeError(arr, '', 'object or array', typeof arr)
  }
  if (!checkValidGPU(f2, end)) {
    manualRun(f2, end, arr)
    return
  }

  const dimensions = getRuntimeDim(ctr, end, mem)

  // external variables to be in the GPU
  const out = { constants: {} }
  out.constants = extern

  const gpuFunction = gpu.createKernel(f, out).setOutput(dimensions)
  const res = gpuFunction() as any
  buildArray(res, ctr, end, mem, extern, arr)
}

function entriesToObject(entries: [string, any][]): any {
  const res = {}
  entries.forEach(([key, value]) => (res[key] = value))
  return res
}

/* tslint:disable-next-line:ban-types */
const kernels: Map<number, Function> = new Map()

export function __clearKernelCache() {
  kernels.clear()
}

export function __createKernelSource(
  end: number[],
  externSource: [string, any][],
  localNames: string[],
  arr: any,
  f: any,
  kernelId: number
) {
  const extern = entriesToObject(externSource)

  const memoizedf = kernels.get(kernelId)
  if (memoizedf !== undefined) {
    return __createKernel(end, extern, memoizedf, arr, f)
  }

  const code = f.toString()
  // We don't need the full source parser here because it's already validated at transpile time.
  const ast = (parse(code, ACORN_PARSE_OPTIONS) as unknown) as es.Program
  const body = (ast.body[0] as es.ExpressionStatement).expression as es.ArrowFunctionExpression
  const newBody = gpuRuntimeTranspile(body, new Set(localNames))
  const kernel = new Function(generate(newBody))

  kernels.set(kernelId, kernel)

  return __createKernel(end, extern, kernel, arr, f)
}
