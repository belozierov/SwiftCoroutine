//
//  FifoQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 09.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct FifoQueue<T> {
    
    private var input = [T]()
    private var output = [T]()
    
    @inlinable internal mutating func insertAtStart(_ item: T) {
        output.append(item)
    }
    
    @inlinable internal mutating func push(_ item: T) {
        input.append(item)
    }
    
    internal mutating func pop() -> T? {
        if let item = output.popLast() { return item }
        switch input.count {
        case 0: return nil
        case 1: return input.popLast()
        default:
            output = input.reversed()
            input.removeAll(keepingCapacity: true)
            return output.popLast()
        }
    }
    
    internal func forEach(_ body: (T) -> Void) {
        (output + input.reversed()).forEach(body)
    }
    
}
