//
//  FifoQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 29.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

struct FifoQueue<T> {
    
    private typealias Pointer = UnsafeMutablePointer<Node>
    
    private struct Node {
        var item: T?
        var state = 0
        var next = 0
        var nextToFree = 0
    }
    
    private var head: Int
    private var tail: Int
    
    private var toFree = 0
    private var accessCount = 0
    
    init() {
        let empty = Pointer.allocate(capacity: 1)
        empty.initialize(to: Node(state: .deleted))
        head = Int(bitPattern: empty)
        tail = Int(bitPattern: empty)
    }
    
    // MARK: - Push
    
    mutating func push(_ item: T) {
        let new = Pointer.allocate(capacity: 1)
        new.initialize(to: Node(item: item))
        atomicAdd(&accessCount, value: 1)
        defer { decreaseAccessCount() }
        while true {
            let tailAddress = tail, tailNode = Pointer(bitPattern: tailAddress)!
            if atomicCAS(&tailNode.pointee.next, expected: 0, desired: Int(bitPattern: new)) {
                return tail = Int(bitPattern: new)
            }
        }
    }
    
    mutating func insertAtStart(_ item: T) {
        let new = Pointer.allocate(capacity: 1)
        new.initialize(to: Node(item: item))
        var empty: Pointer?
        atomicAdd(&accessCount, value: 1)
        while true {
            let headAddress = head
            let headNode = Pointer(bitPattern: headAddress)!
            let nextAddress = headNode.pointee.next
            new.pointee.next = nextAddress
            if nextAddress == 0 {
                if atomicCAS(&headNode.pointee.next, expected: 0, desired: Int(bitPattern: new)) {
                    defer { empty?.deinitialize(count: 1).deallocate() }
                    defer { decreaseAccessCount() }
                    return tail = Int(bitPattern: new)
                }
            } else {
                if empty == nil {
                    empty = .allocate(capacity: 1)
                    empty?.initialize(to: Node(state: .deleted, next: Int(bitPattern: new)))
                }
                if atomicCAS(&head, expected: headAddress, desired: Int(bitPattern: empty)) {
                    return decreaseAccessCount(node: headNode)
                }
            }
        }
    }
    
    // MARK: - Pop
    
    @inlinable mutating func blockingPop() -> T {
        while true { if let item = pop() { return item } }
    }
    
    mutating func pop() -> T? {
        atomicAdd(&accessCount, value: 1)
        while true {
            let headAddress = head
            let headNode = Pointer(bitPattern: headAddress)!
            let nextAddress = headNode.pointee.next
            if let nextNode = Pointer(bitPattern: nextAddress) {
                if tail != headAddress, atomicCAS(&head, expected: headAddress, desired: nextAddress) {
                    defer {
                        if atomicExchange(&nextNode.pointee.state, with: .deleted) == 0 {
                            nextNode.pointee.item = nil
                        }
                        decreaseAccessCount(node: headNode)
                    }
                    return nextNode.pointee.item
                }
            } else {
                decreaseAccessCount()
                return nil
            }
        }
    }
    
    // MARK: - ForEach
    
    internal mutating func forEach(_ body: (T) -> Void) {
        atomicAdd(&accessCount, value: 1)
        forEach(head, nextPath: \.next) { node in
            if atomicUpdate(&node.pointee.state, transform: {
                $0 == .deleted ? .deleted : .reading
            }).new == .reading {
                node.pointee.item.map(body)
                if atomicUpdate(&node.pointee.state, transform: {
                    $0 == .deleted ? .deleted : 0
                }).new == .deleted {
                    node.pointee.item = nil
                }
            }
        }
        decreaseAccessCount()
    }
    
    private func forEach(_ address: Int, nextPath: KeyPath<Node, Int>, body: (Pointer) -> Void) {
        var address = address
        while let pointer = UnsafeMutablePointer<Node>(bitPattern: address) {
            address = pointer.pointee[keyPath: nextPath]
            body(pointer)
        }
    }
    
    // MARK: - Free
    
    mutating func free() {
        freeNodes(head, nextPath: \.next)
        freeNodes(toFree, nextPath: \.nextToFree)
    }
    
    private mutating func decreaseAccessCount(node: Pointer? = nil) {
        let freeFrom = toFree
        if atomicAdd(&accessCount, value: -1) == 1 {
            if freeFrom != 0, atomicCAS(&toFree, expected: freeFrom, desired: 0) {
                freeNodes(freeFrom, nextPath: \.nextToFree)
            }
            node?.deinitialize(count: 1).deallocate()
        } else if let node = node {
            atomicUpdate(&toFree) {
                node.pointee.nextToFree = $0
                return Int(bitPattern: node)
            }
        }
    }
    
    private func freeNodes(_ address: Int, nextPath: KeyPath<Node, Int>) {
        forEach(address, nextPath: nextPath) { $0.deinitialize(count: 1).deallocate() }
    }
    
}

fileprivate extension Int {
    
    static let deleted = 2
    static let reading = 1
    
}
