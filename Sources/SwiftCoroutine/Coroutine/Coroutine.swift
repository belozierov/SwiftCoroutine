//
//  Coroutine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 22.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation
#if SWIFT_PACKAGE
import CCoroutine
#endif

class Coroutine {
    
    static let pool = Pool(creator: Coroutine.init)
    
    static func new(block: @escaping () throws -> Void, resumer: @escaping Resumer) -> Coroutine {
        let coroutine = pool.pop()
        coroutine.resumer = resumer
        coroutine.setBlock { [unowned coroutine, unowned pool] in
            try? block()
            coroutine.free()
            pool.push(coroutine)
        }
        return coroutine
    }
    
    typealias Block = () -> Void
    typealias Resumer = (@escaping Block) -> Void
    private typealias Environment = UnsafeMutablePointer<Int32>
    
    private let block = UnsafeMutablePointer<Block>.allocate(capacity: 1)
    private let stack = UnsafeMutableRawPointer.allocate(byteCount: .stackSize, alignment: 4096)
    private let returnPoint = Environment.allocate(capacity: .environmentSize)
    private let resumePoint = Environment.allocate(capacity: .environmentSize)
    var resumer: Resumer?, onSuspend: Block?
    
    @inline(__always) func setBlock(_ block: @escaping Block) {
        self.block.initialize {
            block()
            longjmp(self.returnPoint, 1)
        }
    }
    
    // MARK: - Start
    
    @inline(__always) func start() {
        resumer?(_start)
    }
    
    private func _start() {
        Thread.current.currentCoroutine = self
        __start(returnPoint, stack.advanced(by: .stackSize), block) {
            $0?.assumingMemoryBound(to: Block.self).pointee()
        }
        Thread.current.currentCoroutine = nil
        onSuspend?()
    }
    
    // MARK: - Resume
    
    @inline(__always) func resume() {
        resumer?(_resume)
    }
    
    private func _resume() {
        Thread.current.currentCoroutine = self
        __save(returnPoint, resumePoint)
        Thread.current.currentCoroutine = nil
        onSuspend?()
    }
    
    // MARK: - Suspend
    
    @inline(__always) func suspend() {
        __save(resumePoint, returnPoint)
    }
    
    func suspendAndResume(with resumer: @escaping Resumer) {
        self.resumer = resumer
        onSuspend = { [unowned self] in
            self.onSuspend = nil
            self.resume()
        }
        suspend()
    }
    
    // MARK: - Free
    
    @inline(__always) func free() {
        block.deinitialize(count: 1)
        onSuspend = nil
        resumer = nil
    }
    
    deinit {
        free()
        stack.deallocate()
        returnPoint.deallocate()
        resumePoint.deallocate()
    }
    
}

extension Coroutine: Hashable {
    
    static func == (lhs: Coroutine, rhs: Coroutine) -> Bool {
        lhs.stack == rhs.stack
    }
    
    func hash(into hasher: inout Hasher) {
        stack.hash(into: &hasher)
    }
    
}

extension Int {
    
    fileprivate static let stackSize = 128 * 1024
    fileprivate static let environmentSize = MemoryLayout<jmp_buf>.size
    
}

extension Thread {
    
    var currentCoroutine: Coroutine? {
        get { threadDictionary.value(forKey: #function) as? Coroutine }
        set { threadDictionary.setValue(newValue, forKey: #function) }
    }
    
}
