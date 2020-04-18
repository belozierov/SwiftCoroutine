//
//  AtomicTuple.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 17.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct AtomicTuple {
    
    internal typealias Tuple = (Int32, Int32)
    private var atomic = AtomicInt()
    
    internal var value: Tuple {
        get { unsafeBitCast(atomic.value, to: Tuple.self) }
        set { atomic.value = unsafeBitCast(newValue, to: Int.self) }
    }
    
    @discardableResult
    internal mutating func update(_ transform: (Tuple) -> Tuple) -> (old: Tuple, new: Tuple) {
        let (old, new) = atomic.update {
            let tuple = unsafeBitCast($0, to: Tuple.self)
            return unsafeBitCast(transform(tuple), to: Int.self)
        }
        return (unsafeBitCast(old, to: Tuple.self), unsafeBitCast(new, to: Tuple.self))
    }
    
    @discardableResult
    internal mutating func update(keyPath: WritableKeyPath<Tuple, Int32>, with value: Int32) -> Int32 {
        update {
            var tuple = $0
            tuple[keyPath: keyPath] = value
            return tuple
        }.old[keyPath: keyPath]
    }
    
}
