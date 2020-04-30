//
//  BlockingFifoQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 29.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

struct BlockingFifoQueue<T> {
    
    private struct Node {
        let item: T
        var next = 0
        var nextToFree = 0
    }
    
    private let condition = PsxCondition()
    private var waiting = 0
    
    private var input = 0
    private var output = 0
    
    private var toFree = 0
    private var popAccessCount = 0
    
    @inlinable mutating func push(_ item: T) {
        push(item, to: &input)
        signalIfNeeded()
    }
    
    @inlinable mutating func insertAtStart(_ item: T) {
        push(item, to: &output)
        signalIfNeeded()
    }
    
    private func push(_ item: T, to address: UnsafeMutablePointer<Int>) {
        let pointer = UnsafeMutablePointer<Node>.allocate(capacity: 1)
        pointer.initialize(to: Node(item: item))
        atomicUpdate(address) {
            pointer.pointee.next = $0
            return Int(bitPattern: pointer)
        }
    }
    
    private mutating func signalIfNeeded() {
        if atomicExchange(&waiting, with: -1) == 1 {
            condition.lock()
            condition.signal()
            condition.unlock()
        }
    }
    
    // MARK: - Pop
    
//    internal mutating func pop2() -> T? {
//        atomicAdd(&popAccessCount, value: 1)
//        let node = popNode()
//        defer { finishPop(with: node) }
//        return node?.pointee.item
//    }
//    
//    private mutating func popNode() -> UnsafeMutablePointer<Node>? {
//        if let item = popOutput() { return item }
//        condition.lock()
//        if let item = popOutput() ?? reverseAndPop() {
//            condition.unlock()
//            return item
//        }
//        condition.unlock()
//        return popOutput()
//    }
    
    // MARK: - Remove first
    
    internal mutating func pop() -> T {
        atomicAdd(&popAccessCount, value: 1)
        let node = removeFirstNode()
        defer { finishPop(with: node) }
        return node.pointee.item
    }
    
    private mutating func removeFirstNode() -> UnsafeMutablePointer<Node> {
        if let item = popOutput() { return item }
        condition.lock()
        while true {
            if let item = popOutput() ?? reverseAndPop() {
                condition.signal()
                condition.unlock()
                return item
            }
            if atomicExchange(&waiting, with: 1) != -1 { condition.wait() }
            atomicStore(&waiting, value: 0)
        }
    }
    
    // MARK: - ForEach
    
    internal mutating func forEach(_ body: (T) -> Void) {
        atomicAdd(&popAccessCount, value: 1)
        condition.lock()
        forEach(input, nextPath: \.next) { body($0.pointee.item) }
        forEach(output, nextPath: \.next) { body($0.pointee.item) }
        condition.unlock()
        finishPop(with: nil)
    }
    
    private func forEach(_ address: Int, nextPath: KeyPath<Node, Int>, body: (UnsafeMutablePointer<Node>) -> Void) {
        var address = address
        while let pointer = UnsafeMutablePointer<Node>(bitPattern: address) {
            address = pointer.pointee[keyPath: nextPath]
            body(pointer)
        }
    }
    
    // MARK: - Output
    
    private mutating func popOutput() -> UnsafeMutablePointer<Node>? {
        if output == 0 { return nil }
        var pointer: UnsafeMutablePointer<Node>?
        atomicUpdate(&output) {
            pointer = UnsafeMutablePointer(bitPattern: $0)
            return pointer?.pointee.next ?? 0
        }
        return pointer
    }
    
    private mutating func reverseAndPop() -> UnsafeMutablePointer<Node>? {
        let old = atomicExchange(&input, with: 0)
        guard var node = UnsafeMutablePointer<Node>(bitPattern: old) else { return nil }
        var nextAddress = node.pointee.next
        node.pointee.next = 0
        while let next = UnsafeMutablePointer<Node>(bitPattern: nextAddress) {
            nextAddress = next.pointee.next
            next.pointee.next = Int(bitPattern: node)
            node = next
        }
        atomicStore(&output, value: node.pointee.next)
        return node
    }
    
    private mutating func finishPop(with node: UnsafeMutablePointer<Node>?) {
        if atomicAdd(&popAccessCount, value: -1) == 1 {
            node?.deinitialize(count: 1).deallocate()
            if toFree != 0 { freeNodes(&toFree, nextPath: \.nextToFree) }
        } else {
            atomicUpdate(&toFree) {
                node?.pointee.nextToFree = $0
                return Int(bitPattern: node)
            }
        }
    }
    
    // MARK: - Free
    
    internal mutating func free() {
        condition.free()
        freeNodes(&input, nextPath: \.next)
        freeNodes(&output, nextPath: \.next)
    }
    
    private func freeNodes(_ address: UnsafeMutablePointer<Int>, nextPath: KeyPath<Node, Int>) {
        forEach(atomicExchange(address, with: 0), nextPath: nextPath) {
            $0.deinitialize(count: 1).deallocate()
        }
    }
    
}
