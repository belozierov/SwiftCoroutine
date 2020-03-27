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
import Darwin

internal final class CoroutineContext {
    
    internal let haveGuardPage: Bool
    internal let stackSize: Int
    private let stack: UnsafeMutableRawPointer
    private let returnEnv: UnsafeMutablePointer<Int32>
    internal var block: (() -> Void)?
    
    internal init(stackSize: Int, guardPage: Bool = true) {
        haveGuardPage = guardPage
        self.stackSize = stackSize
        if guardPage {
            stack = .allocate(byteCount: stackSize + .pageSize, alignment: .pageSize)
            mprotect(stack, .pageSize, PROT_READ)
        } else {
            stack = .allocate(byteCount: stackSize, alignment: .pageSize)
        }
        returnEnv = .allocate(capacity: .environmentSize)
    }
    
    @inlinable internal var stackTop: UnsafeMutableRawPointer {
        .init(stack + stackSize)
    }
    
    // MARK: - Start
    
    @inlinable internal func start() -> Bool {
       __start(returnEnv, stackTop, Unmanaged.passUnretained(self).toOpaque()) {
           _longjmp(Unmanaged<CoroutineContext>
               .fromOpaque($0!)
               .takeUnretainedValue()
               .performBlock(), .finished)
       } == .finished
    }
    
    private func performBlock() -> UnsafeMutablePointer<Int32> {
        block?()
        block = nil
        return returnEnv
    }
    
    // MARK: - Operations
    
    internal struct SuspendData {
        let env: UnsafeMutablePointer<Int32>
        var sp: UnsafeMutableRawPointer!
    }
    
    @inlinable internal func resume(from env: UnsafeMutablePointer<Int32>) -> Bool {
        __save(env, returnEnv, .suspended) == .finished
    }
    
    @inlinable internal func suspend(to data: UnsafeMutablePointer<SuspendData>) {
        __suspend(data.pointee.env, &data.pointee.sp, returnEnv, .suspended)
    }
    
    @inlinable internal func suspend(to env: UnsafeMutablePointer<Int32>) {
        __save(returnEnv, env, .suspended)
    }
    
    deinit {
        returnEnv.deallocate()
        stack.deallocate()
    }
    
}

extension Int32 {
    
    fileprivate static let suspended: Int32 = -1
    fileprivate static let finished: Int32 = 1
    
}

extension CoroutineContext.SuspendData {
    
    internal init() {
        self = .init(env: .allocate(capacity: .environmentSize), sp: nil)
    }
    
}
