//
//  Storage.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 24.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct Storage<T> {
    
    fileprivate typealias Pointer = UnsafeMutablePointer<Node>
    
    fileprivate struct Node {
        var item: T?
        var next = 0
        var nextFree = 0
        var isProcessing = 0
        var tag = 0
        var accessCounter = AtomicTuple()
    }
    
    fileprivate struct FreeNodesQueue {
        private var freeNodes = 0
    }
    
    internal struct Index {
        fileprivate let node: Pointer, tag: Int
    }
    
    private let freeNodes: UnsafeMutablePointer<FreeNodesQueue>
    private var nodes = 0
    
    internal init() {
        freeNodes = .allocate(capacity: 1)
        freeNodes.initialize(to: FreeNodesQueue())
    }
    
    internal var isEmpty: Bool {
        var stack = self.nodes
        while let node = Pointer(bitPattern: stack) {
            if node.pointee.accessCounter.value.0 == 1 { return false }
            stack = node.pointee.next
        }
        return true
    }
    
    // MARK: - Append
    
    @discardableResult internal mutating func append(_ item: T) -> Index {
        let pointer = freePointer()
        pointer.pointee.item = item
        pointer.pointee.accessCounter.value.0 = 1
        return Index(node: pointer, tag: pointer.pointee.tag)
    }
    
    private mutating func freePointer() -> Pointer {
        var pointer: Pointer!
        while true {
            if let free = freeNodes.pointee.popFree() {
                pointer?.deinitialize(count: 1).deallocate()
                return free
            } else if pointer == nil {
                pointer = Pointer.allocate(capacity: 1)
                pointer.initialize(to: Node())
            }
            let address = nodes
            pointer.pointee.next = address
            if atomicCAS(&nodes, expected: address, desired: Int(bitPattern: pointer)) {
                return pointer
            }
        }
    }
    
    // MARK: - Remove
    
    @discardableResult internal func remove(_ index: Index) -> T? {
        while true {
            if atomicCAS(&index.node.pointee.tag, expected: index.tag, desired: index.tag + 1) {
                break
            } else if index.node.pointee.tag != index.tag {
                return nil
            }
        }
        defer { if index.node.pointee.remove() { freeNodes.pointee.pushFree(index.node) } }
        return index.node.pointee.item
    }
    
    internal func removeAll() {
        var next = Pointer(bitPattern: nodes)
        while let node = next {
            next = Pointer(bitPattern: node.pointee.next)
            if node.pointee.remove() { freeNodes.pointee.pushFree(node) }
        }
    }
    
    // MARK: - Iterator
    
    internal func forEach(_ block: (T) -> Void) {
        var next = Pointer(bitPattern: nodes)
        while let node = next {
            next = Pointer(bitPattern: node.pointee.next)
            guard node.pointee.access() else { continue }
            node.pointee.item.map(block)
            if node.pointee.deaccess() { freeNodes.pointee.pushFree(node) }
        }
    }
    
    // MARK: - Free
    
    internal func free() {
        var address = nodes
        while let node = Pointer(bitPattern: address) {
            address = node.pointee.next
            node.deinitialize(count: 1).deallocate()
        }
        freeNodes.deallocate()
    }
    
}

fileprivate extension Storage.Node {
    
    mutating func access() -> Bool {
        accessCounter.update { (hasValue, counter) in
            if hasValue == 0 { return (0, counter) }
            return (1, counter + 1)
        }.old.0 == 1
    }
    
    mutating func deaccess() -> Bool {
        accessCounter.update { ($0.0, max(0, $0.1 - 1)) }.old == (0, 1)
    }
    
    mutating func remove() -> Bool {
        accessCounter.update { (0, $0.1) }.old == (1, 0)
    }
    
}

fileprivate extension Storage.FreeNodesQueue {
    
    typealias Pointer = Storage.Pointer
    
    mutating func popFree() -> Pointer? {
        while true {
            let address = freeNodes
            guard let node = Pointer(bitPattern: address) else { return nil }
            if atomicExchange(&node.pointee.isProcessing, with: 1) == 1 { continue }
            defer { node.pointee.isProcessing = 0 }
            if atomicCAS(&freeNodes, expected: address, desired: node.pointee.nextFree) {
                node.pointee.nextFree = 0
                return node
            }
        }
    }
    
    mutating func pushFree(_ pointer: Pointer) {
        pointer.pointee.item = nil
        while true {
            let address = freeNodes
            pointer.pointee.nextFree = address
            if atomicCAS(&freeNodes, expected: address, desired: Int(bitPattern: pointer)) {
                return
            }
        }
    }
    
}
