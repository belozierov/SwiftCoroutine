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

#if os(Linux)
import Glibc
#else
import Darwin
#endif

internal final class CoroutineContext {
    
    internal let haveGuardPage: Bool
    internal let stackSize: Int
    private let stack: UnsafeMutableRawPointer
    private let returnEnv: UnsafeMutableRawPointer
    internal var block: (() -> Void)?
    
    internal init(stackSize: Int, guardPage: Bool = true) {
        self.stackSize = stackSize
        returnEnv = .allocate(byteCount: .environmentSize, alignment: 16)
        haveGuardPage = guardPage
        if guardPage {
            stack = .allocate(byteCount: stackSize + .pageSize, alignment: .pageSize)
            mprotect(stack, .pageSize, PROT_READ)
        } else {
            stack = .allocate(byteCount: stackSize, alignment: .pageSize)
        }
    }
    
    @inlinable internal var stackTop: UnsafeMutableRawPointer {
        .init(stack + stackSize)
    }
    
    // MARK: - Start
    
    @inlinable internal func start() -> Bool {
       __start(returnEnv, stackTop, Unmanaged.passUnretained(self).toOpaque()) {
           __longjmp(Unmanaged<CoroutineContext>
               .fromOpaque($0!)
               .takeUnretainedValue()
               .performBlock(), .finished)
       } == .finished
    }
    
    private func performBlock() -> UnsafeMutableRawPointer {
        block?()
        block = nil
        return returnEnv
    }
    
    // MARK: - Operations
    
    internal struct SuspendData {
        let env: UnsafeMutableRawPointer
        var sp: UnsafeMutableRawPointer!
    }
    
    @inlinable internal func resume(from env: UnsafeMutableRawPointer) -> Bool {
        __save(env, returnEnv, .suspended) == .finished
    }
    
    @inlinable internal func suspend(to data: UnsafeMutablePointer<SuspendData>) {
        __suspend(data.pointee.env, &data.pointee.sp, returnEnv, .suspended)
    }
    
    @inlinable internal func suspend(to env: UnsafeMutableRawPointer) {
        __save(returnEnv, env, .suspended)
    }
    
    deinit {
        if haveGuardPage {
            mprotect(stack, .pageSize, PROT_READ | PROT_WRITE)
        }
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
        self = .init(env: .allocate(byteCount: .environmentSize, alignment: 16), sp: nil)
    }
    
}
