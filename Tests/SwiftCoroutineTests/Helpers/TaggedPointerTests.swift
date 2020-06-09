//
//  TaggedPointerTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 08.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class TaggedPointerTests: XCTestCase {
    
    struct Tag: OptionSet {
        let rawValue: Int
        static let tag1 = Tag(rawValue: 1 << 0)
        static let tag2 = Tag(rawValue: 1 << 1)
        static let tag3 = Tag(rawValue: 1 << 2)
    }
    
    func testPointer() {
        let pointer = UnsafeMutablePointer<Int>.allocate(capacity: 1)
        defer { pointer.deallocate() }
        var tagged = TaggedPointer<Tag, Int>()
        tagged.pointer = pointer
        tagged[.tag1] = true
        tagged[.tag3] = true
        XCTAssertTrue(tagged[[.tag1, .tag2]])
        XCTAssertTrue(tagged[[.tag1, .tag3]])
        XCTAssertTrue(tagged[.tag1])
        XCTAssertFalse(tagged[.tag2])
        tagged[.tag1] = false
        XCTAssertFalse(tagged[.tag1])
        XCTAssertFalse(tagged[[.tag1, .tag2]])
        XCTAssertTrue(tagged[[.tag1, .tag2, .tag3]])
        XCTAssertEqual(tagged.pointer, pointer)
        XCTAssertEqual(tagged.pointerAddress, Int(bitPattern: pointer))
        tagged.pointer = nil
        XCTAssertNil(tagged.pointer)
        XCTAssertFalse(tagged[.tag2])
        XCTAssertTrue(tagged[[.tag1, .tag2, .tag3]])
        XCTAssertEqual(tagged.pointerAddress, 0)
        let pointer2 = UnsafeMutablePointer<Int>.allocate(capacity: 1)
        defer { pointer2.deallocate() }
        tagged.pointer = pointer2
        XCTAssertEqual(tagged.pointer, pointer2)
    }
    
    func testCounter() {
        var tagged = TaggedPointer<Tag, Int>()
        tagged.counter = 1
        tagged[.tag3] = true
        XCTAssertEqual(tagged.counter, 1)
        tagged.counter = 2
        XCTAssertEqual(tagged.counter, 2)
        XCTAssertTrue(tagged[.tag3])
    }
    
}
