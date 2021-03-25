import { GPU } from 'gpu.js'
import { TypeError } from '../utils/rttc'
import { parse } from 'acorn'
import { generate } from 'astring'
import * as es from 'estree'
import { gpuRuntimeTranspile } from './transfomer'
import { ACORN_PARSE_OPTIONS } from '../constants'

// Heuristic : Only use GPU if array is bigger than this
const MAX_SIZE = 200

// helper function to build array output, modifies and returns res
function buildArray(arr: any, ctr: any, end: any, mem: any, ext: any, res: any) {
  console.log('Building array')
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
      res[i] = buildArray(arr[i], ctr, end, mem.slice(1), ext, res[i])
    }
  } else if (typeof cur === 'string' && cur in ext) {
    const v = ext[cur]
    res[v] = buildArray(arr[v], ctr, end, mem.slice(1), ext, res[v])
  } else if (typeof cur === 'number') {
    res[cur] = buildArray(arr, ctr, end, mem.slice(1), ext, res[cur])
  } else {
    // TODO: handle this properly
    throw 'Index is not number, counter or external variable'
  }

  console.log(res)

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
// function manualRun(f: any, end: any, res: any) {
//   function build() {
//     for (let i = 0; i < end[0]; i++) {
//       res[i] = f(i)
//     }
//     return
//   }

//   function build2D() {
//     for (let i = 0; i < end[0]; i = i + 1) {
//       for (let j = 0; j < end[1]; j = j + 1) {
//         res[i][j] = f(i, j)
//       }
//     }
//     return
//   }

//   function build3D() {
//     for (let i = 0; i < end[0]; i = i + 1) {
//       for (let j = 0; j < end[1]; j = j + 1) {
//         for (let k = 0; k < end[2]; k = k + 1) {
//           res[i][j][k] = f(i, j, k)
//         }
//       }
//     }
//     return
//   }

//   if (end.length === 1) return build()
//   if (end.length === 2) return build2D()
//   return build3D()
// }

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
  console.log('Checking array')
  const endMap = {}
  for (let i = 0; i < ctr.length; i++) {
    endMap[ctr[i]] = end[i]
  }

  let ok = true
  let arrQ = [arr]

  for (let i = 0; i < mem.length - 1; i++) {
    const m = mem[i]
    console.log(m)
    if (typeof m === 'number') {
      // current level of arrays need to be of at least length m
      const newArrQ = []
      for (let a of arrQ) {
        if (!Array.isArray(a) || a.length < m) {
          ok = false
          break
        }
        newArrQ.push(a[m])
      }
      arrQ = newArrQ
    } else if (ctr.includes(m)) {
      // current level of arrays need to be at least length endMap[m]
      const newArrQ = []
      for (let a of arrQ) {
        if (!Array.isArray(a) || a.length < endMap[m]) {
          ok = false
          break
        }
        for (let i = 0; i < endMap[m]; i++) {
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

  for (let a of arrQ) {
    console.log('Checking last level')
    if (!Array.isArray(a)) {
      ok = false
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
  console.log('In __createKernel')
  const gpu = new GPU()

  if (!checkArr(arr, ctr, end, mem, extern)) {
    console.log('Array check failed')
    throw new TypeError(arr, '', 'object or array', typeof arr)
  }
  console.log('Array check passed')

  if (!checkValidGPU(f2, end)) {
    console.log('Manual run')
    // manualRun(f2, end, arr)
    // return
    console.log('Skip manual run for now')
  }

  const dimensions = getRuntimeDim(ctr, end, mem)
  console.log(dimensions)

  // external variables to be in the GPU
  const out = { constants: {} }
  out.constants = extern

  const gpuFunction = gpu.createKernel(f, out).setOutput(dimensions)
  console.log(f)
  const res = gpuFunction() as any
  console.log('GPU.js called')
  console.log(res)
  console.log(arr)
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
  ctr: string[],
  end: number[],
  mem: (string | number)[],
  externSource: [string, any][],
  localNames: string[],
  arr: any,
  f: any,
  kernelId: number
) {
  console.log('In __createKernelSource')
  const extern = entriesToObject(externSource)

  const memoizedf = kernels.get(kernelId)
  if (memoizedf !== undefined) {
    return __createKernel(ctr, end, mem, extern, memoizedf, arr, f)
  }

  const code = f.toString()
  // We don't need the full source parser here because it's already validated at transpile time.
  const ast = (parse(code, ACORN_PARSE_OPTIONS) as unknown) as es.Program
  const body = (ast.body[0] as es.ExpressionStatement).expression as es.ArrowFunctionExpression
  const newBody = gpuRuntimeTranspile(body, new Set(localNames), ctr, mem)
  const kernel = new Function(generate(newBody))

  kernels.set(kernelId, kernel)

  return __createKernel(ctr, end, mem, extern, kernel, arr, f)
}
