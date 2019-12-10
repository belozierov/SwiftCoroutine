//
//  CoroutineContext.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

class CoroutineContext {
    
    typealias Block = () -> Void
    private let stack: UnsafeMutableRawBufferPointer
    private let returnPoint, resumePoint: UnsafeMutablePointer<Int32>
    
    @inline(__always) init(stackSizeInPages: Int) {
        stack = .allocate(byteCount: stackSizeInPages * .pageSize, alignment: .pageSize)
        returnPoint = .allocate(capacity: .environmentSize)
        resumePoint = .allocate(capacity: .environmentSize)
    }
    
    @inline(__always) func start(block: @escaping Block) -> Bool {
        withUnsafePointer(to: { [unowned self] in
            block()
            longjmp(self.returnPoint, .finished)
        }, start)
     }
    
    private func start(with block: UnsafePointer<Block>) -> Bool {
        __start(returnPoint, stack.baseAddress?.advanced(by: stack.count), block) {
            $0?.assumingMemoryBound(to: Block.self).pointee()
        } == .finished
    }
    
    @inline(__always) func resume() -> Bool {
        __save(returnPoint, resumePoint, -1) == .finished
    }
    
    @inline(__always) func suspend() {
        __save(resumePoint, returnPoint, -1)
    }
    
    deinit {
        stack.deallocate()
        returnPoint.deallocate()
        resumePoint.deallocate()
    }
    
}

extension Int {
    
    fileprivate static let pageSize = sysconf(_SC_PAGESIZE)
    fileprivate static let environmentSize = MemoryLayout<jmp_buf>.size
    
}

extension Int32 {
    
    fileprivate static let finished: Int32 = 1
    
}
