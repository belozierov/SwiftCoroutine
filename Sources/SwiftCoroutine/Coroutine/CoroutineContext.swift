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

#if os(macOS)
import Darwin

fileprivate typealias ResumePoint = UnsafeMutablePointer<Int32>
fileprivate let pagesize = _SC_PAGESIZE
#else
import Glibc

fileprivate typealias ResumePoint = UnsafeMutablePointer<__jmp_buf_tag>
fileprivate let pagesize = Int32(_SC_PAGESIZE)
#endif

class CoroutineContext {
    
    typealias Block = () -> Void
    
    let haveGuardPage: Bool
    private let stack: UnsafeMutableRawBufferPointer
    private let returnPoint, resumePoint: ResumePoint
    private var block: Block?
    
    init(stackSize: Int, guardPage: Bool = true) {
        haveGuardPage = guardPage
        if guardPage {
            stack = .allocate(byteCount: stackSize + .pageSize, alignment: .pageSize)
            mprotect(stack.baseAddress, .pageSize, PROT_READ)
        } else {
            stack = .allocate(byteCount: stackSize, alignment: .pageSize)
        }
        returnPoint = .allocate(capacity: .environmentSize)
        resumePoint = .allocate(capacity: .environmentSize)
    }
    
    // MARK: - Start
    
    @inlinable func start(block: @escaping Block) -> Bool {
        self.block = block
        return __start(returnPoint, stackStart,
                       Unmanaged.passUnretained(self).toOpaque()) {
            longjmp(Unmanaged<CoroutineContext>
                .fromOpaque($0!)
                .takeUnretainedValue()
                .performBlock(), .finished)
        } == .finished
     }
    
    private func performBlock() -> ResumePoint {
        block?()
        block = nil
        return returnPoint
    }
    
    // MARK: - Operations
    
    @inlinable func resume() -> Bool {
        __save(returnPoint, resumePoint, -1) == .finished
    }
    
    @inlinable func suspend() {
        __save(resumePoint, returnPoint, -1)
    }
    
    // MARK: - Stack
    
    @inlinable var stackSize: Int {
        stack.count - (haveGuardPage ? .pageSize : 0)
    }
    
    @inlinable var stackStart: UnsafeRawPointer {
        .init(stack.baseAddress!.advanced(by: stack.count))
    }
    
    deinit {
        stack.deallocate()
        returnPoint.deallocate()
        resumePoint.deallocate()
    }
    
}

extension Int {
    
    static let pageSize = Int(sysconf(pagesize))
    fileprivate static let environmentSize = MemoryLayout<jmp_buf>.size
    
}

extension Int32 {
    
    fileprivate static let finished: Int32 = 1
    
}
