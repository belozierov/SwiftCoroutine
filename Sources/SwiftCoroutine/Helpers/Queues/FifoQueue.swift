//
//  FifoQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 29.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct FifoQueue<T> {
    
    private typealias Pointer = UnsafeMutablePointer<Node>
    
    private struct Node: QueueNode {
        
        private(set) var item: T?
        private(set) var using = 0
        var next = 0
        var nextToFree = 0
        
        @inlinable mutating func startUsing() -> Bool {
            atomicUpdate(&using) { $0 > 0 ? $0 + 1 : 0 }.new > 0
        }
        
        @inlinable mutating func endUsing() {
            if atomicAdd(&using, value: -1) == 1 { item = nil }
        }
        
    }
    
    private var head, tail: Int
    private var eraser = QueueNodeEraser<Node>()
    
    internal init() {
        let empty = Pointer.allocate(capacity: 1)
        empty.initialize(to: Node())
        head = Int(bitPattern: empty)
        tail = Int(bitPattern: empty)
    }
    
    // MARK: - Push
    
    internal mutating func push(_ item: T) {
        let new = Pointer.allocate(capacity: 1)
        new.initialize(to: Node(item: item, using: 1))
        eraser.startAccess()
        defer { eraser.endAccess() }
        while true {
            let tailAddress = tail, tailNode = Pointer(bitPattern: tailAddress)!
            if atomicCAS(&tailNode.pointee.next, expected: 0, desired: Int(bitPattern: new)) {
                return tail = Int(bitPattern: new)
            }
        }
    }
    
    internal mutating func insertAtStart(_ item: T) {
        let new = Pointer.allocate(capacity: 1)
        new.initialize(to: Node(item: item, using: 1))
        var empty: Pointer?
        eraser.startAccess()
        while true {
            let headAddress = head
            let headNode = Pointer(bitPattern: headAddress)!
            let nextAddress = headNode.pointee.next
            new.pointee.next = nextAddress
            if nextAddress == 0 {
                if atomicCAS(&headNode.pointee.next, expected: 0, desired: Int(bitPattern: new)) {
                    defer { empty?.deinitialize(count: 1).deallocate() }
                    defer { eraser.endAccess() }
                    return tail = Int(bitPattern: new)
                }
            } else {
                if empty == nil {
                    empty = .allocate(capacity: 1)
                    empty?.initialize(to: Node(next: Int(bitPattern: new)))
                }
                if atomicCAS(&head, expected: headAddress, desired: Int(bitPattern: empty)) {
                    headNode.pointee.next = Int(bitPattern: new)
                    return eraser.endAccess(headNode)
                }
            }
        }
    }
    
    // MARK: - Pop
    
    @inlinable internal mutating func blockingPop() -> T {
        while true { if let item = pop() { return item } }
    }
    
    internal mutating func pop() -> T? {
        eraser.startAccess()
        while true {
            let headAddress = head
            let headNode = Pointer(bitPattern: headAddress)!
            let nextAddress = headNode.pointee.next
            if let nextNode = Pointer(bitPattern: nextAddress) {
                if tail != headAddress, atomicCAS(&head, expected: headAddress, desired: nextAddress) {
                    defer { eraser.endAccess(headNode) }
                    defer { nextNode.pointee.endUsing() }
                    return nextNode.pointee.item
                }
            } else {
                eraser.endAccess()
                return nil
            }
        }
    }
    
    // MARK: - ForEach
    
    internal mutating func forEach(_ body: (T) -> Void) {
        var items = [T]()
        eraser.startAccess()
        var address = head
        while let node = Pointer(bitPattern: address) {
            defer { address = node.pointee.next }
            guard node.pointee.startUsing() else { continue }
            node.pointee.item.map { items.append($0) }
            node.pointee.endUsing()
        }
        eraser.endAccess()
        items.forEach(body)
    }
    
    // MARK: - Free
    
    internal mutating func free() {
        var address = head
        while let node = Pointer(bitPattern: address) {
            address = node.pointee.next
            node.deinitialize(count: 1).deallocate()
        }
        eraser.free()
    }
    
}
