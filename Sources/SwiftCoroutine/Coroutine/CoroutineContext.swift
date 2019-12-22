//
//  CoroutineContext.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#if SWIFT_PACKAGE
import CCoroutine
#endif
import Foundation

class CoroutineContext {
    
    typealias Block = () -> Void
    
    let haveGuardPage: Bool
    private let stack: UnsafeMutableRawBufferPointer
    private let returnPoint, resumePoint: UnsafeMutablePointer<Int32>
    
    @inlinable convenience init() {
        self.init(stackSize: Int(SIGSTKSZ), guardPage: true)
    }
    
    @inlinable convenience init(stackSizeInPages pages: Int, guardPage: Bool = true) {
        let stackSize = max(Int(MINSIGSTKSZ), pages * .pageSize)
        self.init(stackSize: stackSize, guardPage: true)
    }
    
    private init(stackSize: Int, guardPage: Bool) {
        haveGuardPage = guardPage
        stack = .allocate(byteCount: stackSize, alignment: .pageSize)
        if guardPage { mprotect(stack.baseAddress, .pageSize, PROT_READ) }
        returnPoint = .allocate(capacity: .environmentSize)
        resumePoint = .allocate(capacity: .environmentSize)
    }
    
    @inlinable var stackSize: Int {
        stack.count - (haveGuardPage ? .pageSize : 0)
    }
    
    @inlinable var stackStart: UnsafeRawPointer {
        .init(stack.baseAddress!.advanced(by: stack.count))
    }
    
    @inlinable func start(block: @escaping Block) -> Bool {
        var blockRef: Block! = block
        return withUnsafePointer(to: { [unowned(unsafe) self] in
            blockRef()
            blockRef = nil
            longjmp(self.returnPoint, .finished)
        }, start)
     }
    
    private func start(with block: UnsafePointer<Block>) -> Bool {
        __start(returnPoint, stackStart, block) {
            $0?.assumingMemoryBound(to: Block.self).pointee()
        } == .finished
    }
    
    @inlinable func resume() -> Bool {
        __save(returnPoint, resumePoint, -1) == .finished
    }
    
    @inlinable func suspend() {
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
