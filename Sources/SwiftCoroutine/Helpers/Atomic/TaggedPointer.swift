//
//  TaggedPointer.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 07.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

internal struct TaggedPointer<Tag: OptionSet, Element> where Tag.RawValue == Int {
    
    internal var rawValue = 0
    
    // MARK: - tags
    
    @inlinable internal subscript(tag: Tag) -> Bool {
        get { rawValue & tag.rawValue != 0 }
        set { newValue ? (rawValue |= tag.rawValue) : (rawValue &= ~tag.rawValue) }
    }
    
    // MARK: - pointer
    
    @inlinable internal var pointer: UnsafeMutablePointer<Element>? {
        get { UnsafeMutablePointer(bitPattern: pointerAddress) }
        set { rawValue = Int(bitPattern: newValue) | (rawValue & 7) }
    }
    
    @inlinable internal var pointerAddress: Int {
        rawValue & ~7
    }
    
    // MARK: - counter
    
    @inlinable internal var counter: Int32 {
        get { getCounter() }
        set { setCounter(newValue) }
    }
    
    private func getCounter() -> Int32 {
        withUnsafeBytes(of: rawValue) {
            ($0.baseAddress! + 2).assumingMemoryBound(to: Int32.self).pointee
        }
    }
    
    private mutating func setCounter(_ counter: Int32) {
        withUnsafeMutableBytes(of: &rawValue) {
            ($0.baseAddress! + 2).assumingMemoryBound(to: Int32.self).pointee = counter
        }
    }
    
}
