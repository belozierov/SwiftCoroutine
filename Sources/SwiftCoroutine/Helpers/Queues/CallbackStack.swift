//
//  CallbackStack.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 09.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct CallbackStack<T> {
    
    private typealias Pointer = UnsafeMutablePointer<Node>
    
    private struct Node {
        let callback: (T) -> Void
        var next = 0
    }
    
    private var rawValue = 0
    
    private init(rawValue: Int) {
        self.rawValue = rawValue
    }
    
    @inlinable internal init(isFinished: Bool = false) {
        rawValue = isFinished ? -1 : 0
    }
    
    @inlinable internal var isEmpty: Bool { rawValue <= 0 }
    @inlinable internal var isClosed: Bool { rawValue == -1 }
    
    @inlinable internal mutating func append(_ callback: @escaping (T) -> Void) -> Bool {
        var pointer: Pointer!
        while true {
            let address = rawValue
            if address < 0 {
                pointer?.deinitialize(count: 1).deallocate()
                return false
            } else if pointer == nil {
                pointer = .allocate(capacity: 1)
                pointer.initialize(to: Node(callback: callback))
            }
            pointer.pointee.next = address
            if atomicCAS(&rawValue, expected: address, desired: Int(bitPattern: pointer)) {
                return true
            }
        }
    }
    
    @inlinable internal mutating func close() -> Self? {
        let old = atomicExchange(&rawValue, with: -1)
        return old > 0 ? CallbackStack(rawValue: old) : nil
    }
    
    @inlinable internal func finish(with result: T) {
        var address = rawValue
        while address > 0, let pointer = Pointer(bitPattern: address) {
            address = pointer.pointee.next
            pointer.pointee.callback(result)
            pointer.deinitialize(count: 1).deallocate()
        }
    }
    
}
