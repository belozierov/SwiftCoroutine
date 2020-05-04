//
//  QueueNodeEraser.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal protocol QueueNode {
    
    var nextToFree: Int { get set }
    
}

internal struct QueueNodeEraser<T: QueueNode> {
    
    private(set) var accessCount = 0
    private var toFree = 0
    var isFinished = 0
    
    @inlinable internal mutating func startAccess() {
        if isFinished != 0 { print("startAccess error") }
        atomicAdd(&accessCount, value: 1)
    }
    
    internal mutating func endAccess(_ node: UnsafeMutablePointer<T>? = nil) {
        let freeFrom = toFree
        if atomicAdd(&accessCount, value: -1) == 1 {
            if freeFrom != 0, atomicCAS(&toFree, expected: freeFrom, desired: 0) {
                freeNodes(freeFrom)
            }
            node?.deinitialize(count: 1).deallocate()
        } else if let node = node {
            add(node)
        }
        if isFinished != 0 { print("endAccess error") }
    }
    
    @inlinable internal mutating func add(_ node: UnsafeMutablePointer<T>) {
        atomicUpdate(&toFree) {
            node.pointee.nextToFree = $0
            return Int(bitPattern: node)
        }
    }
    
    private func freeNodes(_ address: Int) {
        var address = address
        while let node = UnsafeMutablePointer<T>(bitPattern: address) {
            address = node.pointee.nextToFree
            node.deinitialize(count: 1).deallocate()
        }
    }
    
}
