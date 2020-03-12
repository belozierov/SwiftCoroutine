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

final class CoroutineContext {
    
    let haveGuardPage: Bool
    let stackSize: Int
    private let stack: UnsafeMutableRawPointer
    private let returnEnv: UnsafeMutablePointer<Int32>
    var block: (() -> Void)?
    
    init(stackSize: Int, guardPage: Bool = true) {
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
    
    @inlinable var stackTop: UnsafeMutableRawPointer {
        .init(stack + stackSize)
    }
    
    // MARK: - Start
    
    @inlinable func start() -> Bool {
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
    
    @inlinable func resume(from env: UnsafeMutablePointer<Int32>) -> Bool {
        __save(env, returnEnv, .suspended) == .finished
    }
    
    @inlinable func suspend(to env: UnsafeMutablePointer<__CoroutineEnvironment>) {
        __suspend(env, returnEnv, .suspended)
    }
    
    @inlinable func suspend(to env: UnsafeMutablePointer<Int32>) {
        __save(returnEnv, env, .suspended)
    }
    
    deinit {
        returnEnv.deallocate()
        stack.deallocate()
    }
    
}

extension Int32 {
    
    fileprivate static let suspended: Int32 = -1
    fileprivate static let finished: Int32 = -1
    
}
